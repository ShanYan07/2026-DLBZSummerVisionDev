from maix import camera, display, image, nn, app
import time

# YOLO 模型与摄像头（与 ver1 一致）
detector = nn.YOLOv5(model="/root/models/model-288821.maixcam/model_288821.mud", dual_buff=True)
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()

# 目标类别与运动参数
TARGET_LABEL = "silver_ball"
BASE_SPEED = 0           # 基准速度（原地转向时由 PID 差速驱动）
SPEED_MIN = -150
SPEED_MAX = 150
DEAD_ZONE_PX = 12        # 中心死区半宽（像素），|x_offset| 小于此值时不转向

# PID 参数（根据实车可微调）
PID_KP = 0.12
PID_KI = 0.002
PID_KD = 0.08
INTEGRAL_LIMIT = 80.0
MAX_D_CONTRIB = 25.0     # 微分项最大贡献，防止 reset 后 D 项爆炸
DT_MIN = 1e-4
DT_MAX = 0.08            # 限制单帧 dt，避免卡顿后微分突变

# 卡尔曼滤波参数（越大越平滑，但跟随越慢）
KALMAN_Q_POS = 6.0       # 过程噪声：位置
KALMAN_Q_VEL = 35.0      # 过程噪声：速度
KALMAN_R_MEAS = 90.0     # 测量噪声（YOLO 抖动）
OUTLIER_JUMP_PX = 55     # 单帧跳变超过此值视为野值，仅用预测
MAX_MISS_FRAMES = 12     # 连续丢失检测后重置滤波器
MAX_SPEED_STEP = 10.0    # 每帧轮速最大变化量，抑制输出突变


class PID:
    """位置式 PID，输入为银球中心相对画面中心的 x 方向偏差（像素）。"""

    def __init__(self, kp, ki, kd, integral_limit=INTEGRAL_LIMIT):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.last_error = 0.0
        self.warmup = True

    def compute(self, error, dt):
        dt = clamp(dt, DT_MIN, DT_MAX)

        # reset 后首帧不算微分，避免 (error-0)/dt 尖峰
        if self.warmup:
            self.warmup = False
            self.last_error = error
            return self.kp * error

        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit

        derivative = (error - self.last_error) / dt
        self.last_error = error

        d_term = clamp(self.kd * derivative, -MAX_D_CONTRIB, MAX_D_CONTRIB)
        return self.kp * error + self.ki * self.integral + d_term

    def clear_integral(self):
        """进入死区时只清积分，保留 last_error 防止离开死区时 D 项突变。"""
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.warmup = True


class KalmanPosVel:
    """一维位置-速度卡尔曼滤波，平滑银球中心 x 坐标。"""

    def __init__(self, q_pos, q_vel, r_meas, outlier_jump, max_miss):
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r = r_meas
        self.outlier_jump = outlier_jump
        self.max_miss = max_miss

        self.x = 0.0
        self.v = 0.0
        self.p00 = 400.0
        self.p01 = 0.0
        self.p11 = 400.0
        self.ready = False
        self.miss_count = 0

    def reset(self):
        self.x = 0.0
        self.v = 0.0
        self.p00 = 400.0
        self.p01 = 0.0
        self.p11 = 400.0
        self.ready = False
        self.miss_count = 0

    def _predict(self, dt):
        self.x += self.v * dt

        fp00 = self.p00 + dt * (self.p01 + self.p01 + dt * self.p11)
        fp01 = self.p01 + dt * self.p11
        fp11 = self.p11

        self.p00 = fp00 + self.q_pos
        self.p01 = fp01
        self.p11 = fp11 + self.q_vel

    def predict_only(self, dt):
        if not self.ready:
            return None

        self._predict(dt)
        self.miss_count += 1
        self.p00 += self.r * 0.35
        self.p11 += self.q_vel * 0.5

        if self.miss_count > self.max_miss:
            self.reset()
            return None
        return self.x

    def update(self, measurement, dt):
        if not self.ready:
            self.x = measurement
            self.v = 0.0
            self.ready = True
            self.miss_count = 0
            return self.x

        self._predict(dt)

        innovation = measurement - self.x
        if abs(innovation) > self.outlier_jump:
            self.miss_count += 1
            if self.miss_count > self.max_miss:
                self.reset()
            return self.x

        s = self.p00 + self.r
        k0 = self.p00 / s
        k1 = self.p01 / s

        self.x += k0 * innovation
        self.v += k1 * innovation

        p00_old = self.p00
        p01_old = self.p01
        p11_old = self.p11

        self.p00 = (1.0 - k0) * p00_old
        self.p01 = (1.0 - k0) * p01_old
        self.p11 = p11_old - k1 * p01_old

        self.miss_count = 0
        return self.x


class SpeedSlewLimiter:
    """限制轮速每帧变化量，避免 PID 输出剧烈跳动。"""

    def __init__(self, max_step):
        self.max_step = max_step
        self.left = 0.0
        self.right = 0.0

    def apply(self, target_left, target_right):
        self.left = slew_toward(self.left, target_left, self.max_step)
        self.right = slew_toward(self.right, target_right, self.max_step)
        return self.left, self.right

    def reset(self):
        self.left = 0.0
        self.right = 0.0


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def slew_toward(current, target, max_step):
    delta = target - current
    if delta > max_step:
        return current + max_step
    if delta < -max_step:
        return current - max_step
    return target


def find_silver_ball(objs):
    """从检测结果中选取置信度最高的 silver_ball。"""
    best_obj = None
    best_score = 0.0

    for obj in objs:
        label = detector.labels[obj.class_id]
        if label != TARGET_LABEL:
            continue
        if obj.score > best_score:
            best_score = obj.score
            best_obj = obj

    return best_obj


def apply_dead_zone(x_offset):
    """中心死区：偏差足够小时视为已对准，避免抖动。"""
    if abs(x_offset) <= DEAD_ZONE_PX:
        return 0
    if x_offset > 0:
        return x_offset - DEAD_ZONE_PX
    return x_offset + DEAD_ZONE_PX


def calc_wheel_speeds(x_offset, pid, dt):
    """
    根据 x 方向偏差计算左右轮速度。
    x_offset > 0 表示银球在画面中心右侧，小车应右转（左轮快、右轮慢）。
    允许负速度（反转），范围 [-150, 150]。
    """
    control_error = apply_dead_zone(x_offset)
    if control_error == 0:
        pid.clear_integral()
        return 0.0, 0.0, 0.0

    turn = pid.compute(control_error, dt)
    left_speed = clamp(BASE_SPEED + turn, SPEED_MIN, SPEED_MAX)
    right_speed = clamp(BASE_SPEED - turn, SPEED_MIN, SPEED_MAX)
    return left_speed, right_speed, turn


pid = PID(PID_KP, PID_KI, PID_KD)
kalman = KalmanPosVel(KALMAN_Q_POS, KALMAN_Q_VEL, KALMAN_R_MEAS, OUTLIER_JUMP_PX, MAX_MISS_FRAMES)
speed_limiter = SpeedSlewLimiter(MAX_SPEED_STEP)
image_center_x = detector.input_width() // 2
last_time = time.perf_counter()

while not app.need_exit():
    now = time.perf_counter()
    dt = now - last_time
    last_time = now

    img = cam.read()
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45)

    # 绘制画面中心竖线及死区边界
    img.draw_line(image_center_x, 0, image_center_x, detector.input_height() - 1, color=image.COLOR_GREEN)
    dead_left = image_center_x - DEAD_ZONE_PX
    dead_right = image_center_x + DEAD_ZONE_PX
    img.draw_line(dead_left, 0, dead_left, detector.input_height() - 1, color=image.COLOR_WHITE)
    img.draw_line(dead_right, 0, dead_right, detector.input_height() - 1, color=image.COLOR_WHITE)

    silver_ball = find_silver_ball(objs)

    for obj in objs:
        color = image.COLOR_YELLOW if detector.labels[obj.class_id] == TARGET_LABEL else image.COLOR_RED
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=color)
        label = detector.labels[obj.class_id]
        msg = f"{label}: {obj.score:.2f}"
        img.draw_string(obj.x, obj.y, msg, color=color)

    if silver_ball is not None:
        raw_center_x = silver_ball.x + silver_ball.w // 2
        ball_center_y = silver_ball.y + silver_ball.h // 2
        filtered_center_x = kalman.update(raw_center_x, dt)
        filtered_center_x = int(round(filtered_center_x))
        x_offset = filtered_center_x - image_center_x

        img.draw_cross(raw_center_x, ball_center_y, color=image.COLOR_BLUE, size=6)
        img.draw_cross(filtered_center_x, ball_center_y, color=image.COLOR_GREEN, size=10)

        in_dead_zone = abs(x_offset) <= DEAD_ZONE_PX
        left_target, right_target, turn = calc_wheel_speeds(x_offset, pid, dt)
        left_speed, right_speed = speed_limiter.apply(left_target, right_target)

        zone_tag = " [dead]" if in_dead_zone else ""
        print(
            f"raw={raw_center_x:3d} filt={filtered_center_x:3d} "
            f"dx={x_offset:+4d}px | turn={turn:+6.2f} | "
            f"left={left_speed:+4.0f} right={right_speed:+4.0f}{zone_tag}"
        )

        speed_msg = f"L:{left_speed:+.0f} R:{right_speed:+.0f}"
        img.draw_string(4, 4, speed_msg, color=image.COLOR_WHITE)
        offset_msg = f"dx:{x_offset:+d} f:{filtered_center_x}"
        img.draw_string(4, 20, offset_msg, color=image.COLOR_WHITE)
    else:
        predicted_x = kalman.predict_only(dt)

        if predicted_x is not None:
            # 短时丢检：用卡尔曼预测继续控制，不 reset PID
            filtered_center_x = int(round(predicted_x))
            x_offset = filtered_center_x - image_center_x
            left_target, right_target, turn = calc_wheel_speeds(x_offset, pid, dt)
            left_speed, right_speed = speed_limiter.apply(left_target, right_target)
            print(
                f"lost track, predict filt={filtered_center_x:3d} dx={x_offset:+4d}px | "
                f"turn={turn:+6.2f} | left={left_speed:+4.0f} right={right_speed:+4.0f}"
            )
            offset_msg = f"pred dx:{x_offset:+d}"
            img.draw_string(4, 4, "track lost", color=image.COLOR_WHITE)
            img.draw_string(4, 20, offset_msg, color=image.COLOR_WHITE)
        else:
            pid.reset()
            left_speed, right_speed = speed_limiter.apply(0.0, 0.0)
            print(f"silver_ball not found | left={left_speed:+4.0f} right={right_speed:+4.0f}")
            img.draw_string(4, 4, "no silver_ball", color=image.COLOR_WHITE)

    disp.show(img)

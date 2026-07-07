from maix import camera, display, image, nn, app
import time
detector = nn.YOLOv5(model="/root/models/0706DataSet_Cam2Model/model_289006.mud", dual_buff = True)

cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()

while not app.need_exit():
    t_start = time.perf_counter() # 记录开始时间（秒）
    img = cam.read()
    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
    disp.show(img)

    t_end = time.perf_counter()   # 记录结束时间
    
    # 计算一帧消耗的时间，并换算成 FPS
    duration = t_end - t_start
    if duration > 0:
        fps = 1.0 / duration
        print(f"FPS: {fps:.2f}")
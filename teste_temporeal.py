from ultralytics import YOLO
import cv2

model = YOLO('train/weights/best.pt')

results = model.track(source="0", show = True, save_crop = True, save_txt = True, conf = 0.7)

print(results)
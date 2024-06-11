from ultralytics import YOLO
import torch
from ultralytics import NAS
#import super_gradients



if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == '__main__':
    #model = YOLO('yolov8n.pt').load('runs/detect/train47/weights/best.pt')  # load a pretrained YOLOv8n classification model
    model = YOLO('yolov8n.pt')
    # Load a COCO-pretrained YOLO-NAS-s model
    #model = NAS('yolo_nas_s.pt')
    model.train(
        data='dataset_2802/data.yaml', 
        epochs=1000, 
        patience=50,
        batch= -1,
        device=0,
        lrf = 0.1,
        #augment=True, 
        #hsv_h = .015,
        #hsv_s = .7,
        #hsv_v = .4,
        #degrees = .4,
        #translate = .3,
        #scale = .5,
        #shear = .01,
        #flipud = .3,
        #fliplr = .5,
        #mixup = .5
        )


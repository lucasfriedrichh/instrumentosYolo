import cv2
import os
from PIL import Image
from ultralytics import YOLO

# Função para dividir vídeo em frames
def video_para_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    return frame_count

# Função para rodar YOLO em cada frame e salvar resultados
def yolo_nos_frames(frames_dir, output_dir, model):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    folder_name = os.path.basename(frames_dir)
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        img = Image.open(frame_path)
        results = model(img)
        
        # Salvar resultados em .txt
        txt_filename = os.path.join(output_dir, f'{os.path.splitext(frame_file)[0]}.txt')
        with open(txt_filename, 'w') as f:
            for *box, conf, cls in results.xyxy[0].tolist():
                x1, y1, x2, y2 = map(int, box)
                # Salvar nome da pasta no lugar da classe
                f.write(f'{folder_name} {x1} {y1} {x2} {y2} {conf:.2f}\n')
                
        # Opcional: salvar imagens com bounding boxes desenhadas (descomente para salvar)
        # results.save(output_dir)

# Caminho do vídeo de entrada e diretórios de saída
video_path = 'caminho do video'
frames_dir = 'frames'
output_dir = 'resultados'

# Dividir vídeo em frames
frame_count = video_para_frames(video_path, frames_dir)
print(f'{frame_count} frames extraídos do vídeo.')


model = YOLO('train/weights/best.pt')

# Rodar YOLO nos frames e salvar resultados
yolo_nos_frames(frames_dir, output_dir, model)
print('Detecção YOLO concluída e resultados salvos.')

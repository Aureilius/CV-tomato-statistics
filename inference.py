import cv2
from ultralytics import YOLO
import random

def weight_calculation(x1,y1, x2, y2):
    pixel2cm = 0.03
    pi = 3.14
    x = abs(x2-x1)
    y = abs(y2-y1)

    x *= pixel2cm
    y *= pixel2cm
    res = int(f'{pi * x ** 3/ 5:.0f}')
    return 120 if res > 1000 else res

def process_video_with_tracking(model, input_video_path, show_video=True, save_video=True, output_video_path='output_video.mp4'):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception('Error: Could not open video file.')
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    weights = {}

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=640, verbose=False, tracker='botsort.yaml')

        if results[0].boxes.id != None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box,id in zip(boxes,ids):
                random.seed(int(id))
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                weight = weight_calculation(box[0], box[1], box[2], box[3])
                if id not in weights:
                    weights[id] = weight
                else:
                    weights[id] = max(weights[id],weight)
                    #weights[id] = (weights[id]+weight)/2

                cv2.rectangle(frame, (box[0], box[1]), (box[2],box[3],), color, 2)
                cv2.putText(
                    frame,
                    f"id {id}, weight: ~{weights[id]:.0f}g",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
        
        if save_video:
            out.write(frame)
        
        if show_video:
            frame = cv2.resize(frame, (0,0), fx=0.75, fy=0.75)
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    #print(f'{sum(weights.values()):.2f}')
    cap.release()
    if save_video:
        out.release()

    cv2.destroyAllWindows()
    return weights.values()
    

model = YOLO('runs/detect/train2/weights/best.pt')
model.fuse()
process_video_with_tracking(model, 'testing.mp4', show_video=True, save_video=True, output_video_path='output_video2.mp4')
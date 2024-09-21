import cv2
import torch
import mediapipe as mp
import numpy as np


from physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from Loadtemporal_data_test import Normaliztion, ToTensor
from torchvision import transforms

# 1. Video Capture and Preprocessing
cap = cv2.VideoCapture(0)  # Open the default webcam

# Face detection using MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Model parameters
image_size = (160, 128, 128)  # Adjust based on your model and input
gra_sharp = 2.0  # Adjust as needed

# 2. Model Loading and Setup
model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=image_size)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Transformations for preprocessing
transform = transforms.Compose([Normaliztion(), ToTensor()])

# 3. Process Video and Extract rPPG
rppg_signal = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection using MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]

            # Preprocessing
            face_roi_processed = cv2.resize(face_roi, (image_size[1], image_size[2]))
            face_roi_processed = face_roi_processed.transpose((2, 0, 1))  # Channel first
            face_roi_processed = face_roi_processed[np.newaxis, ...]  # Add batch dimension
            sample = {'video_x': face_roi_processed}
            sample = transform(sample)
            input_tensor = sample['video_x'].unsqueeze(0).float().cuda()  # To tensor and GPU

            # Model inference
            with torch.no_grad():
                rppg, _, _, _ = model(input_tensor, gra_sharp)

            # Append rPPG signal (adjust if needed based on model output)
            rppg_signal.extend(rppg.squeeze().tolist())

            # (Optional) Display or process the rPPG signal in real-time
            # ...

            # Draw bounding box on the frame (optional)
            cv2.rectangle(frame, bbox, (255, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Further process or visualize the rppg_signal
# ...
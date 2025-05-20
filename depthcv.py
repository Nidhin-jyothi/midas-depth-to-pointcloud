import cv2
import torch
import numpy as np

# Load lightweight MiDaS model and transforms
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to('cpu')

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()

    # Normalize and convert to 8-bit for colormap
    depth_min = prediction.min()
    depth_max = prediction.max()
    depth_norm = (prediction - depth_min) / (depth_max - depth_min + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    # Show webcam and depth map
    cv2.imshow('Webcam Frame', frame)
    cv2.imshow('Depth Map', depth_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

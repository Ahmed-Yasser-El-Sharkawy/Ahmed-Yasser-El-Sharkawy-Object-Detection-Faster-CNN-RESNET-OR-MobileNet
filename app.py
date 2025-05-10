import streamlit as st
import cv2
import torch
import os
import numpy as np
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load Models
@st.cache_resource
def load_model(backbone_name, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if backbone_name == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load('fasterrcnnResnet.pth', map_location=device))
    elif backbone_name == 'mobilenet':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load('fasterrcnnMobilenet.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

class_names = ['background', 'Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

# Inference Function for Images
def predict_image(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image / 255.0).permute(2, 0, 1).float().unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)[0]
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_names[label]}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

# Inference Function for Videos
def predict_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_tensor = torch.tensor(frame / 255.0).permute(2, 0, 1).float().unsqueeze(0)
        with torch.no_grad():
            output = model(frame_tensor)[0]
        for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
            if score > 0.5:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_names[label]}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frames.append(frame)
    cap.release()
    output_path = 'output_video.mp4'
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return output_path

# Streamlit Interface
st.title('Object Detection - Streamlit Implementation')

model_choice = st.selectbox('Select Model', ['ResNet50', 'MobileNet'])
model = load_model(model_choice.lower(), num_classes=6)

# Image Inference
st.subheader('Image Inference')
uploaded_image = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    image_path = uploaded_image.name
    with open(image_path, 'wb') as f:
        f.write(uploaded_image.read())
    st.image(predict_image(image_path, model), caption='Detected Objects', use_column_width=True)

# Video Inference
st.subheader('Video Inference')
uploaded_video = st.file_uploader('Upload a Video', type=['mp4', 'avi'])

if uploaded_video:
    video_path = uploaded_video.name
    with open(video_path, 'wb') as f:
        f.write(uploaded_video.read())
    output_video_path = predict_video(video_path, model)
    st.video(output_video_path)

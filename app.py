import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import time
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# EfficientNet
@st.cache_resource
def load_model():
    # arch
    model = models.efficientnet_b0()
    # remale last cover
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    # vesa
    model.load_state_dict(torch.load("src/efficientnet_sota.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()
classes = ["funko", "mug", "pen"]

# our interface
st.title("CV2 SOTA Demo")
st.write("Загрузите картинку и модель покажет класс и тепловую карту Grad-CAM.")
st.write("**Environment:** MacBook Pro M4")

uploaded_file = st.file_uploader("Выберите изображение ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Эта картинка", use_container_width=True)

    # preproc
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = tf(image).unsqueeze(0)
    start_time = time.time()

    # preds
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = probs.max(dim=1)

    end_time = time.time()
    latency = (end_time - start_time) * 1000

    st.write(f"Предикт: **{classes[pred.item()]}**")
    st.write(f"Акураси: {conf.item() * 100:.2f}%")
    st.write(f"**Латенси:** {latency:.2f} ms")

    # GradСAM
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred.item())]

    # warmCard
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # on pic
    img_resized = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

    st.write("Grad-CAM:")
    st.write("Модель смотрит на красную область на тепловой карте")
    st.image(visualization, use_container_width=True)
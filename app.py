import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

st.title("Bird vs Drone Detection System")

# CNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# CNN
cnn_model = SimpleCNN()
cnn_model.load_state_dict(torch.load("cnn_final.pth", map_location="cpu"))
cnn_model.eval()

# ResNet
resnet_model = models.resnet50(weights=None)

resnet_model.fc = nn.Sequential(
    nn.Linear(resnet_model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

resnet_model.load_state_dict(torch.load("resnet_final.pth", map_location="cpu"))
resnet_model.eval()

# YOLO
yolo_model = YOLO("best.pt")



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["bird", "drone"]



uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    st.write(f"### CNN Prediction: {classes[pred.item()]} ({conf.item()*100:.2f}%)")


    with torch.no_grad():
        output = resnet_model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    st.write(f"### ResNet Prediction: {classes[pred.item()]} ({conf.item()*100:.2f}%)")


    results = yolo_model(image)

    st.write("### YOLO Detection:")
    st.image(results[0].plot(), caption="YOLO Output")
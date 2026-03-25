import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load("models/bee_model.pth", map_location=device))
model.to(device)
model.eval()

classes = ["bee", "wasp"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("🐝 Bee vs Wasp Classifier")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    confidence = probs[0][pred.item()].item()

    st.success(f"Prediction: {classes[pred.item()]}")
    st.info(f"Confidence: {confidence:.2f}")
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["bee", "wasp"]

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load("models/bee_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    confidence = probs[0][pred.item()].item()

    return classes[pred.item()], confidence

# Test
label, conf = predict("test.jpg")
print(f"Prediction: {label}, Confidence: {conf:.2f}")
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

labels = [
    'Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanocytic Nevi', 'Melanoma', 'Squamous Cell Carcinoma', 'Vascular Lesions'
]

test_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()  
    return model, device

def predict_image(model, image_path, device):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    transformed = test_transforms(image=image)
    image_tensor = transformed['image']  
    image_tensor = image_tensor.unsqueeze(0) 
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # Xác suất
        predicted_class = torch.argmax(outputs, dim=1).cpu().item()  
    
    predicted_label = labels[predicted_class]
    print(f"Predicted Label: {predicted_label}")
    print("Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{labels[i]}: {prob:.4f}")
    
    return predicted_label, probabilities

if __name__ == "__main__":
    model_path = 'Model/full_model.pth'  
    image_path = 'Test/uit.jpg'  # path to image
    
    try:
        model, device = load_model(model_path)
        predict_image(model, image_path, device)
    except Exception as e:
        print(f"Lỗi: {e}")
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
latent_dim=512

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 3x224x224 -> 32x112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 128x28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 256x14x14
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> 512x7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 512x7x7 -> 256x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 128x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64x56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32x112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # -> 3x224x224
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

torch.serialization.add_safe_globals(['__main__.Autoencoder'])

threshold = 0.25

model_path = 'Model/autoencoder_skin.pth' 

# Function to compute reconstruction error for a single image
def get_reconstruction_error(model, image_path, transform, criterion, device):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img)
    img_tensor = augmented['image'].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        recon = model(img_tensor)
        error = criterion(recon, img_tensor).item()
    return error

# Function to test a single image
def test_image(model, image_path, transform, criterion, device, threshold):
    try:
        error = get_reconstruction_error(model, image_path, transform, criterion, device)
        is_anomaly = error > threshold
        result = "Anomaly (Not a skin lesion)" if is_anomaly else "Normal (Skin lesion)"
        print(f"Image: {os.path.basename(image_path)}, Error: {error:.4f}, Result: {result}")
        return error, is_anomaly
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

# Load the model
try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    exit()


criterion = nn.MSELoss()

test_image_path = 'Test/bkl.jpg'
test_image(model, test_image_path, transform, criterion, device, threshold)

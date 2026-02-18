import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import os

# ================= CONFIGURATION (Your Previous Specs) =================
MODEL_PATH = "best_offroad_model.pth"
TEST_IMG_PATH = "image_to_test.png"  
SAVE_PATH = "predicted_mask.png"
IMG_SIZE = 256
NUM_CLASSES = 50  # Matches your training config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL ARCHITECTURE =================
def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        self.down1 = double_conv(3, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = double_conv(128, 64)
        self.out = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        up_2 = self.up2(c3)
        merge2 = torch.cat([up_2, c2], dim=1)
        c4 = self.conv_up2(merge2)
        up_1 = self.up1(c4)
        merge1 = torch.cat([up_1, c1], dim=1)
        c5 = self.conv_up1(merge1)
        return self.out(c5)


def run_test():
    # 1. Load Model
    model = UNet(NUM_CLASSES).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained model file not found!")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Process Image
    if not os.path.exists(TEST_IMG_PATH):
        print(f"Error: Please place an image named '{TEST_IMG_PATH}' in this folder.")
        return

    img = Image.open(TEST_IMG_PATH).convert("RGB")
    original_size = img.size
    img_input = TF.resize(img, (IMG_SIZE, IMG_SIZE))
    img_tensor = TF.to_tensor(img_input).unsqueeze(0).to(DEVICE)

    # 3. Predict
    with torch.no_grad():
        output = model(img_tensor)
        # Select the class with the highest probability
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 4. Save (Multiplied by 5 to make classes visible to the eye)
    result = Image.fromarray((pred_mask * 5).astype(np.uint8))
    result = result.resize(original_size, resample=Image.NEAREST)
    result.save(SAVE_PATH)
    print(f"Prediction saved as {SAVE_PATH}")

if __name__ == "__main__":
    run_test()
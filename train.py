
import os, cv2, torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


DATA_DIR = r"C:\Users\yashm\OneDrive\Desktop\Offroad_Segmentation_Training_Dataset\train"
IMG_SIZE = 256  
BATCH_SIZE = 4  
EPOCHS = 50
NUM_CLASSES = 50 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SegDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = sorted([f for f in os.listdir(os.path.join(root,"images")) if f.endswith(".png")])
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img_path = os.path.join(self.root,"images",name)
        mask_path = os.path.join(self.root,"masks",name)

        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros((IMG_SIZE, IMG_SIZE)).long()

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path,0)
        mask = cv2.resize(mask,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_NEAREST)

        return self.tf(img), torch.from_numpy(mask).long()


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c,out_c,3,padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c,out_c,3,padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.c1 = conv_block(3,64); self.p1 = nn.MaxPool2d(2)
        self.c2 = conv_block(64,128); self.p2 = nn.MaxPool2d(2)
        self.c3 = conv_block(128,256); self.p3 = nn.MaxPool2d(2)
        self.c4 = conv_block(256,512); self.p4 = nn.MaxPool2d(2)
        self.b = conv_block(512,1024)
        self.u1 = nn.ConvTranspose2d(1024,512,2,2); self.c5 = conv_block(1024,512)
        self.u2 = nn.ConvTranspose2d(512,256,2,2); self.c6 = conv_block(512,256)
        self.u3 = nn.ConvTranspose2d(256,128,2,2); self.c7 = conv_block(256,128)
        self.u4 = nn.ConvTranspose2d(128,64,2,2); self.c8 = conv_block(128,64)
        self.out = nn.Conv2d(64,n_classes,1)

    def forward(self,x):
        c1 = self.c1(x); c2 = self.c2(self.p1(c1)); c3 = self.c3(self.p2(c2)); c4 = self.c4(self.p3(c3))
        b = self.b(self.p4(c4))
        u1 = torch.cat([self.u1(b),c4],1); c5 = self.c5(u1)
        u2 = torch.cat([self.u2(c5),c3],1); c6 = self.c6(u2)
        u3 = torch.cat([self.u3(c6),c2],1); c7 = self.c7(u3)
        u4 = torch.cat([self.u4(c7),c1],1); c8 = self.c8(u4)
        return self.out(c8)

def iou_score(pred, mask):
    pred = torch.argmax(pred,1)
    intersection = (pred & mask).float().sum()
    union = (pred | mask).float().sum()
    return (intersection+1e-6)/(union+1e-6)


if __name__ == '__main__':
    print(f"TRAINING STARTED ON: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found. Training on CPU will be slow.")

    dataset = SegDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(DEVICE=="cuda"))

    model = UNet(NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE=="cuda"))

    for e in range(EPOCHS):
        model.train()
        tot_loss = 0
        tot_iou = 0
        
        loop = tqdm(loader, desc=f"Epoch {e+1}/{EPOCHS}")
        for img, mask in loop:
            img, mask = img.to(DEVICE), mask.to(DEVICE)

            with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                pred = model(img)
                loss = loss_fn(pred, mask)

            opt.zero_grad()
            if DEVICE == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            tot_loss += loss.item()
            tot_iou += iou_score(pred, mask).item()
            loop.set_postfix(loss=tot_loss/len(loader), iou=tot_iou/len(loader))

        print(f"Epoch {e+1} Summary | Loss: {tot_loss/len(loader):.4f} | IoU: {tot_iou/len(loader):.4f}")

    torch.save(model.state_dict(), "unet_gpu_model.pth")
    print("TRAINING COMPLETE - Model saved as unet_gpu_model.pth")





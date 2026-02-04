# training/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.cnn_rnn import CNNRNN
from dataset.ocr_dataset import OCRDataset
from utils.charset import CHARS, BLANK_IDX

# -------------------------
# DEVICE
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

# -------------------------
# COLLATE FUNCTION (VERY IMPORTANT FOR CTC)
# -------------------------
def collate_fn(batch):
    images, labels = zip(*batch)

    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)

    return images, labels, label_lengths

# -------------------------
# DATASET & LOADER
# -------------------------
dataset = OCRDataset(
    image_dir="data/synthetic/images",
    labels_file="data/synthetic/labels.txt",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

# -------------------------
# MODEL
# -------------------------
model = CNNRNN(num_classes=len(CHARS) + 1)
model.to(DEVICE)

# -------------------------
# LOSS & OPTIMIZER
# -------------------------
criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# TRAINING LOOP
# -------------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels, label_lengths in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE)

        outputs = model(images)
        outputs = outputs.log_softmax(2)

        batch_size = outputs.size(0)
        time_steps = outputs.size(1)

        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=time_steps,
            dtype=torch.long
        ).to(DEVICE)

        loss = criterion(
            outputs.permute(1, 0, 2),  # (time, batch, classes)
            labels,
            input_lengths,
            label_lengths
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "ocr_model.pth")
print("✅ Model saved as ocr_model.pth")

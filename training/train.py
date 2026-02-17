# training/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from models.cnn_rnn import CNNRNN
from dataset.ocr_dataset import OCRDataset
from utils.charset import CHARS, BLANK_IDX

# -------------------------
# DEVICE
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

# -------------------------
# COLLATE FUNCTION
# -------------------------
def collate_fn(batch):
    images, labels = zip(*batch)

    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)

    return images, labels, label_lengths

# -------------------------
# DATASET
# -------------------------
dataset = OCRDataset(
    image_dir="data/synthetic/images",
    labels_file="data/synthetic/labels.txt",
    transform=transform
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=16,   # stable for CPU
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
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
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# -------------------------
# TRAINING LOOP
# -------------------------
EPOCHS = 15

for epoch in range(EPOCHS):

    # ---- TRAINING ----
    model.train()
    train_loss = 0.0

    for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):

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
            outputs.permute(1, 0, 2),
            labels,
            input_lengths,
            label_lengths
        )

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (important for CTC + LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} - Batch {batch_idx}/{len(train_loader)}")

    train_loss /= len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels, label_lengths in val_loader:

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
                outputs.permute(1, 0, 2),
                labels,
                input_lengths,
                label_lengths
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"- Train Loss: {train_loss:.4f} "
        f"- Val Loss: {val_loss:.4f}"
    )

# -------------------------
# SAVE MODEL
# -------------------------
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "ocr_model.pth")

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
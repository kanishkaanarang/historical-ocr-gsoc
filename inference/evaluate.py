# inference/evaluate.py

import torch
from PIL import Image
import torchvision.transforms as transforms

from models.cnn_rnn import CNNRNN
from utils.charset import CHARS
from utils.ctc_decoder import ctc_greedy_decode
from dataset.ocr_dataset import OCRDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

def main():
    model = CNNRNN(num_classes=len(CHARS) + 1)
    model.load_state_dict(torch.load("ocr_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dataset = OCRDataset(
        image_dir="data/synthetic/images",
        labels_file="data/synthetic/labels.txt",
        transform=transform
    )

    print("\n--- OCR Evaluation (GT vs Prediction) ---\n")

    for i in range(5):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            log_probs = output.log_softmax(2)

        pred = ctc_greedy_decode(log_probs[0])
        gt = "".join([CHARS[c - 1] for c in label])

        print(f"Image {i}")
        print(f"GT   : {gt}")
        print(f"PRED : {pred}")
        print("-" * 30)

if __name__ == "__main__":
    main()

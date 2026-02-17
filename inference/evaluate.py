# inference/evaluate.py
from utils.ctc_decoder import ctc_greedy_decode, ctc_beam_search_decode
import torch
from PIL import Image
import torchvision.transforms as transforms

from models.cnn_rnn import CNNRNN
from utils.charset import CHARS
from utils.ctc_decoder import ctc_greedy_decode
from dataset.ocr_dataset import OCRDataset
from utils.metrics import character_error_rate, word_error_rate
import os
print("Current working directory:", os.getcwd())
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

    val_indices = torch.load("val_indices.pt")
    dataset = torch.utils.data.Subset(dataset, val_indices)
    print("\n--- OCR Evaluation (GT vs Prediction) ---\n")

    total_cer = 0
    total_wer = 0
    num_samples = 20  # Evaluate on first 20 samples (adjust if needed)

    for i in range(num_samples):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            log_probs = output.log_softmax(2)

            log_probs_sample = log_probs[0]

            pred_greedy = ctc_greedy_decode(log_probs_sample)
            pred_beam = ctc_beam_search_decode(log_probs_sample, beam_width=5)

            # Choose which one to evaluate
            pred = pred_beam
            gt = "".join([CHARS[c - 1] for c in label])

        cer = character_error_rate(gt, pred)
        wer = word_error_rate(gt, pred)

        total_cer += cer
        total_wer += wer

        print(f"Image {i}")
        print(f"GT      : {gt}")
        print(f"Greedy  : {pred_greedy}")
        print(f"Beam    : {pred_beam}")
        print(f"PRED : {pred}")
        print(f"CER  : {cer:.4f}")
        print(f"WER  : {wer:.4f}")
        print("-" * 40)

    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples

    print("\n--- Overall Metrics ---")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Character Accuracy: {(1 - avg_cer) * 100:.2f}%")
    print(f"Word Accuracy: {(1 - avg_wer) * 100:.2f}%")

if __name__ == "__main__":
    main()
    
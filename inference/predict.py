# inference/predict.py

import torch
from PIL import Image
import torchvision.transforms as transforms

from models.cnn_rnn import CNNRNN
from utils.charset import CHARS
from utils.ctc_decoder import ctc_greedy_decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("L")
    img = transform(img)
    img = img.unsqueeze(0)  # add batch dimension
    return img

def main():
    model = CNNRNN(num_classes=len(CHARS) + 1)
    model.load_state_dict(torch.load("ocr_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_image = "data/synthetic/images/img_0.png"
    img = load_image(test_image).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        log_probs = output.log_softmax(2)
    print(torch.argmax(log_probs[0], dim=1)[:50])
    decoded = ctc_greedy_decode(log_probs[0])

    print("Predicted text:", decoded)

if __name__ == "__main__":
    main()

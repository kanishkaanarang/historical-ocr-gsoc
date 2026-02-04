# data/generate_synthetic.py
import sys
import os
import random
from PIL import Image, ImageDraw, ImageFont

# -------------------------
# CONFIGURATION
# -------------------------

OUTPUT_DIR = "data/synthetic/images"
LABEL_FILE = "data/synthetic/labels.txt"

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64
FONT_SIZE = 32
NUM_SAMPLES = 2000  # start small, increase later

# Basic character set (can expand later)
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 "

# Path to a system font (adjust if needed)
if sys.platform.startswith("win"):
    FONT_PATH = "C:/Windows/Fonts/arial.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# -------------------------
# UTILITY FUNCTIONS
# -------------------------

def random_text(min_len=3, max_len=10):
    common = "abcdefghijklmnopqrstuvwxyz "
    rare = "0123456789"

    length = random.randint(min_len, max_len)
    text = ""

    for _ in range(length):
        if random.random() < 0.25:
            text += random.choice(rare)
        else:
            text += random.choice(common)

    return text

def create_image(text, font):
    img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (IMAGE_WIDTH - text_width) // 2
    y = (IMAGE_HEIGHT - text_height) // 2

    draw.text((x, y), text, fill=0, font=font)
    return img

# -------------------------
# MAIN SCRIPT
# -------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        print("Could not load font, using default font.")
    font = ImageFont.load_default()


    with open(LABEL_FILE, "w") as f:
        for i in range(NUM_SAMPLES):
            text = random_text()
            img = create_image(text, font)

            filename = f"img_{i}.png"
            img.save(os.path.join(OUTPUT_DIR, filename))

            f.write(f"{filename},{text}\n")

    print(f"Generated {NUM_SAMPLES} synthetic samples.")

if __name__ == "__main__":
    main()

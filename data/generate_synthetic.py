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
LEXICON_FILE = "data/lexicon.txt"

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64
FONT_SIZE = 32
NUM_SAMPLES = 10000

# Path to system font
if sys.platform.startswith("win"):
    FONT_PATH = "C:/Windows/Fonts/arial.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# -------------------------
# LOAD LEXICON
# -------------------------

if not os.path.exists(LEXICON_FILE):
    raise FileNotFoundError(
        "Lexicon file not found. Create data/lexicon.txt first."
    )

with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    WORDS = [w.strip() for w in f.readlines() if w.strip()]

# Character set derived from lexicon

# -------------------------
# TEXT GENERATION
# -------------------------

def generate_text():
    """
    Generate 1–3 word phrase from lexicon.
    Occasionally inject numbers to simulate noise.
    """
    num_words = random.randint(1, 3)
    words = random.choices(WORDS, k=num_words)

    text = " ".join(words)

    # Occasionally insert random digit at end
    if random.random() < 0.2:
        text += str(random.randint(0, 9))

    return text


# -------------------------
# IMAGE CREATION
# -------------------------

def create_image(text, font):
    img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = max((IMAGE_WIDTH - text_width) // 2, 5)
    y = max((IMAGE_HEIGHT - text_height) // 2, 5)

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
        print("Could not load system font. Using default font.")
        font = ImageFont.load_default()

    with open(LABEL_FILE, "w", encoding="utf-8") as f:
        for i in range(NUM_SAMPLES):
            text = generate_text()
            img = create_image(text, font)

            filename = f"img_{i}.png"
            img.save(os.path.join(OUTPUT_DIR, filename))

            f.write(f"{filename},{text}\n")

    print(f"Generated {NUM_SAMPLES} synthetic samples with lexicon-based text.")


if __name__ == "__main__":
    main()
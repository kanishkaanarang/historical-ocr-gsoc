# utils/charset.py

import string

# basic character set (can expand later)
CHARS = string.ascii_lowercase + string.digits + " "

# mapping
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}

BLANK_IDX = 0

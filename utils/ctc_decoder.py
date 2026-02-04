# utils/ctc_decoder.py

import torch
from utils.charset import idx_to_char, BLANK_IDX

def ctc_greedy_decode(log_probs):
    """
    log_probs: Tensor of shape (time, classes)
    """
    preds = torch.argmax(log_probs, dim=1)

    decoded = []
    prev = BLANK_IDX

    for p in preds:
        p = p.item()
        if p != BLANK_IDX and p != prev:
            decoded.append(idx_to_char[p])
        prev = p

    return "".join(decoded)

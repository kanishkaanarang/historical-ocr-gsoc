# utils/ctc_decoder.py

import torch
import math
from utils.charset import CHARS, BLANK_IDX


def ctc_greedy_decode(log_probs):
    """
    Greedy CTC decoding.
    """
    pred_indices = torch.argmax(log_probs, dim=1)

    decoded = []
    prev = None

    for idx in pred_indices:
        idx = idx.item()
        if idx != BLANK_IDX and idx != prev:
            decoded.append(CHARS[idx - 1])
        prev = idx

    return "".join(decoded)


def ctc_beam_search_decode(log_probs, beam_width=5):
    """
    Simple CTC Beam Search Decoder.

    Args:
        log_probs (Tensor): shape (time, num_classes)
        beam_width (int): number of beams to keep

    Returns:
        str: decoded string
    """

    time_steps, num_classes = log_probs.shape

    # Each beam: (prefix_string, accumulated_log_prob, last_char_index)
    beams = [("", 0.0, None)]

    for t in range(time_steps):
        new_beams = []

        for prefix, score, last_char in beams:
            for c in range(num_classes):
                log_p = log_probs[t, c].item()

                new_score = score + log_p

                if c == BLANK_IDX:
                    # blank extends prefix without adding char
                    new_beams.append((prefix, new_score, last_char))
                else:
                    char = CHARS[c - 1]

                    # Avoid repeating same char unless separated by blank
                    if c == last_char:
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + char

                    new_beams.append((new_prefix, new_score, c))

        # Sort beams by score descending
        new_beams.sort(key=lambda x: x[1], reverse=True)

        # Keep top beams
        beams = new_beams[:beam_width]

    # Return best beam
    best_prefix = beams[0][0]

    return best_prefix
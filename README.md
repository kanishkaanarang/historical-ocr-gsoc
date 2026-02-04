# Historical OCR using CNN–BiLSTM with CTC Loss

This project implements an end-to-end Optical Character Recognition (OCR)
pipeline for historical-style text using a CNN–BiLSTM architecture trained
with Connectionist Temporal Classification (CTC) loss.

The goal of this work is to build a clear, extensible OCR baseline that can
later be adapted to Renaissance-era and other historical documents where
labeled data is scarce and text layouts are irregular.

---

## Motivation

Digitizing historical documents is challenging due to noisy images, degraded
paper, irregular fonts, and the lack of large annotated datasets. Traditional
OCR systems often perform poorly on such material.

This project focuses on understanding and implementing the core components
of a modern OCR system, emphasizing interpretability, modular design, and
practical experimentation rather than state-of-the-art performance.

---

## Methodology

The OCR pipeline consists of the following stages:

1. **Synthetic Data Generation**
   - Text strings are programmatically generated and rendered into grayscale
     images using PIL.
   - Weighted character sampling is used to oversample less frequent symbols
     and stabilize early CTC training.

2. **Feature Extraction (CNN)**
   - A convolutional neural network extracts visual features such as strokes
     and character shapes from the input image.

3. **Sequence Modeling (BiLSTM)**
   - CNN feature maps are interpreted as a left-to-right sequence.
   - A bidirectional LSTM models contextual dependencies between characters.

4. **Alignment-Free Training (CTC Loss)**
   - CTC loss enables training without explicit character-level alignment
     between image regions and text labels.

5. **Decoding and Evaluation**
   - Greedy CTC decoding is used for inference.
   - Predictions are compared against ground truth to analyze systematic
     character-level errors.

---

## Data

At the current stage, training is performed entirely on synthetic data.
This approach allows controlled experimentation, avoids licensing issues,
and mirrors common practice in historical OCR research.

The pipeline is designed so that real historical text images can be
integrated later without architectural changes.

---

## Results and Observations

The model successfully learns to produce non-trivial character sequences
and preserves correct character ordering in many cases. Errors are typically
small and structured, such as confusions between visually similar characters
or collapsed repeated symbols, which are expected behaviors in early-stage
CTC-based OCR systems.

These results confirm that the OCR pipeline is functionally correct and
provide a solid foundation for further improvements.

---

## Project Structure

data/ - synthetic data generation and storage
models/ - CNN–BiLSTM OCR model definition
dataset/ - dataset loading and encoding
training/ - training loop with CTC loss
inference/ - prediction and evaluation scripts
utils/ - character set and CTC decoding utilities

---

## Future Work

Planned extensions include:
- Lexicon-aware decoding (e.g. beam search) to reduce linguistically invalid
  predictions
- Support for accented Spanish characters and historical glyph variants
- Integration with text-detection tools (e.g. Hi-SAM) to enable full-page
  historical document transcription

---

## Notes

This repository is intended as a learning-oriented and research-oriented
baseline rather than a production OCR system. Emphasis is placed on clarity,
explainability, and incremental improvement.

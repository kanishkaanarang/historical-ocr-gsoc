# 📜 Historical OCR using CNN–BiLSTM with CTC Loss

An end-to-end Optical Character Recognition (OCR) pipeline for historical-style text using a CNN–BiLSTM architecture trained with Connectionist Temporal Classification (CTC) loss.

This repository serves as a research-oriented baseline for Renaissance and historical document transcription, aligned with HumanAI’s GSoC 2026 OCR proposal.

---

## 🎯 Project Objective

Historical documents pose challenges including:

- Irregular typography  
- Degraded print  
- Rare characters and symbols  
- Limited labeled datasets  

This project builds a modular, explainable OCR pipeline that:

- Learns character recognition directly from images  
- Handles sequence alignment without manual segmentation  
- Provides measurable evaluation metrics  
- Can be extended to Renaissance-era Spanish print  

---

## 🧠 Architecture Overview

Image (64x256 grayscale)
↓
CNN (feature extraction)
↓
BiLSTM (sequence modeling)
↓
Linear projection
↓
CTC Loss
↓
Greedy / Beam Search Decoding
---

### 1️⃣ CNN – Visual Feature Extraction

The convolutional network learns:

- Stroke patterns  
- Character shapes  
- Local spatial structure  

It converts the 2D image into a feature map interpreted as a left-to-right sequence.

---

### 2️⃣ BiLSTM – Context Modeling

The bidirectional LSTM:

- Processes the feature sequence in both directions  
- Captures contextual dependencies between characters  
- Improves recognition of ambiguous shapes  

---

### 3️⃣ CTC Loss – Alignment-Free Training

Connectionist Temporal Classification (CTC):

- Eliminates need for character-level bounding boxes  
- Learns alignment between image features and text  
- Handles variable-length sequences  

This is standard in modern OCR systems.

---

## 🧪 Synthetic Data Generation

Since historical labeled data is scarce, training is performed on synthetic data generated using:

- Weighted character sampling  
- Word-level and phrase-level text  
- Multi-word sequences  
- Randomized lengths  
- Centered rendering using PIL  

This mirrors common practice in historical OCR research.

---

## 📊 Evaluation Metrics

Implemented metrics:

- Character Error Rate (CER)  
- Word Error Rate (WER)  
- Character Accuracy  
- Word Accuracy  

---

## 📈 Current Results (Synthetic Dataset)

Training setup:

- 10,000 synthetic samples  
- 15 epochs  
- CPU training  
- Train/Validation split  

### Performance

| Metric | Value |
|--------|--------|
| Character Accuracy | 95.29% |
| Word Accuracy | 83.33% |
| Average CER | 0.047 |
| Average WER | 0.166 |

### Observed Error Types

- Missing trailing digits  
- Rare character confusion  
- Minor spelling distortions  
- Slight truncation in long phrases  

These behaviors are typical for early-stage CTC-based OCR systems.

---

## 🔍 Decoding Methods

### Greedy Decoding
Selects the most probable character at each timestep.

### Beam Search Decoding
Maintains multiple candidate sequences to improve stability and reduce local errors.

---

## 📂 Project Structure

historical-ocr-gsoc/
│
├── data/ # Synthetic data generation
├── dataset/ # Dataset loading and encoding
├── models/ # CNN–BiLSTM model definition
├── training/ # Training loop with CTC
├── inference/ # Prediction and evaluation
├── utils/ # Charset, CTC decoding, beam search
├── requirements.txt
└── README.md

---

## 🚀 How To Run

### Install dependencies

```bash
pip install -r requirements.txt

Generate synthetic data
python data/generate_synthetic.py

Train model
python -m training.train

Evaluate model
python -m inference.evaluate

```

## Future Work

Planned improvements aligned with GSoC 2026 proposal:

Lexicon-constrained beam search

Weighted learning for rare glyphs

Spanish diacritic support

Image augmentation (noise, blur, distortion)

LLM-based transcription refinement

Integration with page-level text detection (e.g., Hi-SAM)

## Design Philosophy

This repository prioritizes:

Modularity

Reproducibility

Metric-driven evaluation

Research alignment with historical OCR challenges

It is intended as a foundation for expanding toward Renaissance-era Spanish transcription tasks.

---


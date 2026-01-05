# 🧠 Deep Learning Model Zoo

A hands-on deep learning repository implementing **core CNN and RNN architectures** — from **scratch-level fundamentals** to **production-style training pipelines**.

This repository contains **10 carefully designed models (5 CNN + 5 RNN)** built to develop **deep intuition about representation learning, sequence modeling, training dynamics, and architectural trade-offs** — not just to chase accuracy.

---

## 🎯 Project Goals

- Move from **theoretical understanding → real implementation**
- Understand **what happens inside CNNs and RNNs**
- Learn **why architectures fail, improve, or saturate**
- Build **resume-ready, interview-relevant deep learning projects**
- Prepare a strong foundation for **Transformers and Agentic AI**

---

## 🧠 CNN Models (Vision → Representation Learning)

### 1️⃣ CNN From Scratch (No `nn.Conv2d`)
**Dataset:** MNIST  
**Focus:** First principles

**Implemented manually**
- Convolution (sliding window)
- Stride & padding
- ReLU
- Max pooling

**Why it matters**  
You understand *exactly* what Conv2D and pooling layers do internally.

**Resume-ready skill**
> Implemented convolution and pooling operations from scratch and trained a CNN classifier.

---

### 2️⃣ LeNet-5 (Classic CNN)
**Dataset:** MNIST  
**Focus:** Architectural discipline

- Controlled parameter count
- Stable training
- Historical significance

**Why it matters**  
LeNet explains *why CNNs worked even before modern GPUs*.

**Resume-ready skill**
> Implemented and trained LeNet-5 for digit classification.

---

### 3️⃣ Deep CNN (VGG-Style, No Skip Connections)
**Dataset:** CIFAR-10  
**Focus:** Depth vs performance

- Vanishing gradients
- Training slowdown
- Accuracy saturation

**What you observe**
- Deeper ≠ always better
- Optimization becomes harder with depth

**Resume-ready skill**
> Designed and trained a deep CNN inspired by VGG architecture.

---

### 4️⃣ CNN with BatchNorm & Dropout
**Dataset:** CIFAR-10  
**Focus:** Training stability & regularization

- Batch Normalization effects on gradients
- Dropout vs overfitting

**Why it matters**  
This is **real-world CNN engineering**, not toy experimentation.

**Resume-ready skill**
> Improved CNN training stability using Batch Normalization and Dropout.

---

### 5️⃣ CNN as Feature Extractor + Custom Classifier
**Dataset:** CIFAR-10 / Custom images  
**Focus:** Production-style workflow

- Frozen CNN backbone
- Trainable classifier head

**Why it matters**  
This is how **most production vision systems** are built.

**Resume-ready skill**
> Used CNN backbone for feature extraction and trained a custom classification head.

---

## 🧠 RNN Models (Sequence → Memory → Time)

### 6️⃣ Vanilla RNN (Character-Level Language Model)
**Dataset:** Text (names / Shakespeare / code)  
**Focus:** Temporal dynamics

- Hidden state flow
- Time unrolling
- Gradient explosion / vanishing

**Why it matters**  
You *feel* why vanilla RNNs struggle.

**Resume-ready skill**
> Built a character-level language model using a vanilla RNN.

---

### 7️⃣ LSTM for Sequence Prediction
**Dataset:** Time-series / text  
**Focus:** Long-term dependencies

- Input, forget, output gates
- Memory control

**Key insight**  
LSTM = **controlled memory writing**

**Resume-ready skill**
> Implemented and trained an LSTM for sequence modeling.

---

### 8️⃣ GRU vs LSTM Comparison
**Dataset:** Same dataset for fair comparison  
**Focus:** Architectural trade-offs

- Training speed
- Accuracy
- Convergence behavior

**Why it matters**  
This comparison is **frequently asked in interviews**.

**Resume-ready skill**
> Compared GRU and LSTM architectures on sequential data.

---

### 9️⃣ Seq2Seq Encoder–Decoder (No Attention)
**Dataset:** Simple text → text mapping  
**Focus:** Encoder–decoder mechanics

- RNN encoder
- RNN decoder
- Teacher forcing

**Pain point**
- Context vector bottleneck

**Resume-ready skill**
> Implemented a sequence-to-sequence encoder-decoder model.

---

### 🔟 Seq2Seq With Attention (Bahdanau / Luong)
**Focus:** Dynamic memory access

- Alignment scores
- Context vectors
- Improved decoding

**Critical insight**
> Attention = dynamic memory addressing

**Resume-ready skill**
> Enhanced Seq2Seq model with attention mechanism for improved performance.

---

## 🛠️ Tech Stack

- Python
- PyTorch (minimal abstraction where possible)
- NumPy
- Matplotlib (analysis & debugging)

---

## 📂 Suggested Repository Structure

deep-learning-model-zoo/
│
├── cnn/
│ ├── cnn_from_scratch/
│ ├── lenet5/
│ ├── deep_cnn_vgg_style/
│ ├── cnn_batchnorm_dropout/
│ └── cnn_feature_extractor/
│
├── rnn/
│ ├── vanilla_rnn/
│ ├── lstm/
│ ├── gru_vs_lstm/
│ ├── seq2seq_no_attention/
│ └── seq2seq_with_attention/
│
├── utils/
├── experiments/
└── README.md




---

## 🚀 Who This Repository Is For

- ML/DL learners moving from **theory → implementation**
- Engineers preparing for **deep learning interviews**
- Anyone who wants a **strong foundation before Transformers & Agentic AI**

---

## 📌 Future Extensions

- Transformers & self-attention
- Vision Transformers (ViT)
- Training optimization & profiling
- Agentic AI building blocks

---

## ⭐ Final Note

This repository prioritizes **understanding over shortcuts**.  
Every model is built to answer *why* something works — not just *how to run it*.

If you find this useful, feel free to ⭐ the repo.

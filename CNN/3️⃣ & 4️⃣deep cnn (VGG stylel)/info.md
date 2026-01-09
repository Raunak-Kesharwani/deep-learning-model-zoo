 **1-page Deep Learning Training Cheat Sheet** distilled from everything you practiced.  
This is designed to be **fast to scan during experiments**, **interview-safe**, and **competition-ready**.  

---  

# 🧠 Deep Learning Training Cheat Sheet  

*(CNN-focused, practitioner level)*  

---  

## 📌 DATA AUGMENTATION  

**Purpose:** Increase data diversity, not dataset size  

**Key facts**  
  
* Dataset length stays the same  
* Each epoch sees different image variants  
* Stronger than dropout for vision models  

**Standard CIFAR-10 setup**  

```  
RandomCrop(32, padding=4)  
RandomHorizontalFlip(p=0.5)  
```  

**Rules**  

* Apply only to training data  
* Validate/test data must stay clean  

**Mental model**  

> Augmentation teaches *invariance*, not memorization.  

---  

## 📌 DROPOUT (WHERE & HOW)  

**Best placement**  

```  
Linear → BatchNorm → ReLU → Dropout  
```  

**Where to use**  

* Fully Connected layers → ✅  
* Conv blocks → ⚠️ small values only (optional)  

**Types**  

* `Dropout` → FC layers  
* `Dropout2d` → Conv blocks (≤ 0.2)  

**Common mistake**  

* Using too much dropout to fix architectural issues  

**Mental model**  

> Dropout prevents co-adaptation, not feature learning.  

---

## 📌 WEIGHT DECAY (OPTIMIZER DEPENDENT)  

### SGD  

* `5e-4` → standard  
* `1e-3` → strong  
* True L2 regularization  

### Adam / AdamW  

* `1e-4` → standard  
* `1e-5` → weak  
* AdamW decouples decay from gradients  

**Rule**  

> Weight decay values are **not transferable** across optimizers.  

---  

## 📌 COSINE LEARNING RATE SCHEDULING   

**When to use**  

* Medium/deep CNNs  
* ≥ 50 epochs  

**Correct usage**  

```  
T_max = total_epochs  
eta_min = 1e-5  
```

**Why it works**  
* Early epochs → exploration  
* Late epochs → fine convergence  

**Mistake**  

* Using cosine with too few epochs  

**Mental model**  

> Cosine LR improves *final convergence*, not early learning.   

---  

## 📌 OVERFITTING DIAGNOSIS  

### True overfitting  

* Train loss ↓  
* Val loss ↓ then ↑  
* Train acc ↑  
* Val acc ↓ / plateaus   

### Late-epoch overfitting (normal)  

* Val acc stable  
* Val loss increases  
* Model becomes more confident, not more correct  

**Rule**  

> Accuracy plateau + rising val loss = confidence overfitting  

---  

## 📌 EARLY STOPPING (MANDATORY)  

**Why**  

* Best model ≠ last epoch  
* Prevents memorization after convergence  

**Correct approach**  

* Monitor validation loss  
* Save best checkpoint  
* Stop after no improvement (patience ≈ 10)  

**Rule**  

> You don’t fix late overfitting — you stop before it happens.  

---  

## 🧠 MASTER TRAINING RULE  

> **Regularization controls overfitting**  
> **Optimization controls convergence**  
> **Architecture controls performance ceiling**  

Once overfitting is solved:  
➡️ Stop regularizing  
➡️ Improve optimization or architecture  

1️⃣ torch.backends.cudnn.benchmark & deterministic  

These control how PyTorch chooses CUDA kernels for CNN ops.  
 
🔹 torch.backends.cudnn.benchmark  
What it does  

When benchmark=True:  

cuDNN tests multiple convolution algorithms  

Picks the fastest one for your input shape  

Reuses it every iteration  

Mental model  

“Try all possible convolution engines once → remember the fastest”  

When to ENABLE (True)  

✅ CNNs  
✅ Fixed input size (like CIFAR-10: 32×32)  
✅ Training for performance  
✅ Competitions / experiments  

When to DISABLE (False)  

❌ Variable input sizes  
❌ Debugging numerical issues  
❌ Strict reproducibility needed  

Recommendation for you  
torch.backends.cudnn.benchmark = True  

🔹 torch.backends.cudnn.deterministic  
What it does  

Forces cuDNN to:  

Use only deterministic algorithms  

Avoid non-deterministic CUDA kernels  

Mental model  

“Always do the exact same math every run”  

When to ENABLE (True)  

✅ Debugging  
✅ Verifying correctness  
✅ Research reproducibility  
✅ Paper results  

When to DISABLE (False)  

❌ Competitions  
❌ Performance runs  
❌ Large CNN training  

Recommendation for you  
torch.backends.cudnn.deterministic = False  

🧠 Golden rule (memorize)
Phase	                           benchmark	       deterministic  
Debugging	                           ❌	              ✅  
Training	                           ✅	              ❌  
Competition	                           ✅	              ❌  
Paper reproducibility	               ❌	              ✅  


🧠 Final cheat sheet 
| Concept            | Purpose           | Use now?  |  
| ------------------ | ----------------- | --------  |  
| `benchmark`        | Speed             | ✅        |  
| `deterministic`    | Reproducibility   | ❌        |  
| `scheduler.step()` | LR decay          | ✅        |  
| GAP                | Reduce FC overfit | ✅        |  
| Label smoothing    | Generalization    | ✅        |  



| Operation           | You specify     | Used for             |   
| ------------------- | --------------- | -------------------- |  
| `Conv2d`            | kernel, stride  | Feature extraction   |  
| `MaxPool2d`         | kernel, stride  | Local downsampling   |  
| `AvgPool2d`         | kernel, stride  | Local smoothing      |  
| `AdaptiveAvgPool2d` | **output size** | Global summarization |  
| `AdaptiveMaxPool2d` | **output size** | Feature presence     |  
| GAP                 | output=(1,1)    | Classification       |  
| Attention pooling   | output size     | Learned aggregation  |  
| ROI Align           | output size     | Detection            |  

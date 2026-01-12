gradient clipping is already applied :
when trained the model but with 20 epochs there are all meaning less characters 
when trained the model but with 100 character sequence are inheriting meaning but still bad because of long sequnece length 
when trained with 100 epoches but with 10 sequence length got better results more words got meaning becasue of this 

this results bad because in RNN there are not options to maintain the long term and short term memory seprately 
LSTM helps becasue they have seperate states to maintain these both things 
attention works more better becasue in attention mechanism we pay attention to a particular text when attention shift we have rememberace of previous text in memory or we say previous essense of word is used to get some meaning from a sentense.






## 🔹 Vanilla RNN Language Modeling Project

* Implemented a **Vanilla RNN cell from scratch** and trained a character-level language model on Shakespeare.
* Understood how RNNs work by **time unrolling with shared weights** and how the hidden state acts as **short-term memory**.
* Clearly distinguished between:

  * **BPT** (backpropagation in feedforward networks),
  * **BPTT** (full backpropagation through all timesteps),
  * **TBPTT** (carrying hidden-state values while truncating gradients).
* Learned that **hidden state and gradients are independent**:

  * Resetting `h` forgets context,
  * Detaching `h` stops gradient flow but keeps memory.
* Implemented a **practical TBPTT variant** (carry hidden state across batches, detach each batch, reset periodically).
* Learned to **read and debug loss curves**, understanding why loss can decrease while generated text still lacks meaning.
* Identified common RNN failure modes: **vanishing/exploding gradients**, poor sampling, and short context windows.
* Learned correct **target representation** for sequence classification (class indices, not one-hot vectors).
* Understood the difference between **training (teacher forcing)** and **generation (autoregressive sampling)**.
* Gained practical insight into **why vanilla RNNs fail at long-range dependencies**, motivating LSTM, attention, and Transformers.

---
## 🔑 One-Line Takeaway

> *This project taught me how sequence models actually learn, how BPTT and TBPTT work in practice, and why modern architectures replace vanilla RNNs.*
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random

# ===============================
# DEVICE & REPRODUCIBILITY
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device.type} device")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# ===============================
# DATASET: WIKITEXT-2 (BUILT-IN)
# ===============================
from torchtext.datasets import WikiText2
from torchtext.vocab import Vocab
from collections import Counter

def load_wikitext():
    train_iter = WikiText2(split="train")
    return list(train_iter)

raw_text = load_wikitext()

def tokenize(text):
    return text.split()

tokens = []
for line in raw_text:
    tokens.extend(tokenize(line))

vocab = Vocab(
    Counter(tokens),
    specials=["<pad>", "<unk>"]
)
vocab.set_default_index(vocab["<unk>"])

encoded_data = torch.tensor(
    [vocab[token] for token in tokens],
    dtype=torch.long
)

# ===============================
# TRAIN / TEST SPLIT
# ===============================
train_size = int(0.9 * len(encoded_data))
train_data = encoded_data[:train_size]
test_data  = encoded_data[train_size:]

# ===============================
# DATASET CLASS (LANGUAGE MODEL)
# ===============================
sequence_length = 35
batch_size = 64

class LanguageModelDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return x, y

train_dataset = LanguageModelDataset(train_data, sequence_length)
test_dataset  = LanguageModelDataset(test_data, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===============================
# MODEL: LSTM LANGUAGE MODEL
# ===============================
class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

# ===============================
# TRAIN FUNCTION
# ===============================
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch % 200 == 0:
            print(f"loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity

# ===============================
# TEST FUNCTION
# ===============================
def test(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)

    print(
        f"Test Metrics:\n"
        f" Perplexity: {perplexity:.2f}\n"
        f" Avg loss: {avg_loss:.4f}"
    )

    return avg_loss, perplexity

# ===============================
# TRAINING LOOP
# ===============================
epochs = 20
lr = 3e-4

model = LSTM_Model(len(vocab)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

train_losses, train_ppls = [], []
test_losses, test_ppls = [], []

best_ppl = float("inf")

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}\n-------------------------------")

    tr_loss, tr_ppl = train(
        train_loader, model, loss_fn, optimizer, device
    )
    te_loss, te_ppl = test(
        test_loader, model, loss_fn, device
    )

    train_losses.append(tr_loss)
    train_ppls.append(tr_ppl)
    test_losses.append(te_loss)
    test_ppls.append(te_ppl)

    if te_ppl < best_ppl:
        best_ppl = te_ppl
        torch.save(model.state_dict(), "LSTM_WikiText2.pth")

print("Training Complete")

# ===============================
# PLOTTING
# ===============================
from matplotlib import pyplot as plt

epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(6,4))
plt.plot(epochs_range, train_ppls, label="Train Perplexity")
plt.plot(epochs_range, test_ppls, label="Test Perplexity")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("LSTM Language Model Perplexity")
plt.legend()
plt.grid(True)

plt.figure(figsize=(6,4))
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Language Model Loss")
plt.legend()
plt.grid(True)

plt.show()

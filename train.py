import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import SymbolDecoder

PAD = 4  # padding token index

sequences = []
labels = []

with open("data.txt", "r") as f:
    for line in f:
        seq_str, label_str = line.strip().split("],")
        seq = eval(seq_str + "]")   # convert "[0,1,2" → [0,1,2]
        label = int(label_str)
        sequences.append(seq)
        labels.append(label)


max_len = max(len(s) for s in sequences)

padded_sequences = []
for seq in sequences:
    padded = seq + [PAD] * (max_len - len(seq))
    padded_sequences.append(padded)

inputs = torch.tensor(padded_sequences, dtype=torch.long)
targets = torch.tensor(labels, dtype=torch.long)


model = SymbolDecoder(vocab_size=5, embed_dim=16, hidden_dim=32, num_classes=4)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 2000
loss_history = []
acc_history = []

print("Training started...")

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(inputs)          # (batch, num_classes)
    loss = loss_fn(logits, targets)

    loss.backward()
    optimizer.step()

    # Track loss
    loss_history.append(loss.item())

    # Track accuracy
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == targets).float().mean().item()
    acc_history.append(accuracy)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Acc: {accuracy*100:.2f}%")

print("Training complete.")

torch.save(model.state_dict(), "model.pth")
print("Saved model.pth")

plt.figure(figsize=(8,4))
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss.png")
print("Saved loss.png")

plt.figure(figsize=(8,4))
plt.plot(acc_history)
plt.title("Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("accuracy.png")
print("Saved accuracy.png")
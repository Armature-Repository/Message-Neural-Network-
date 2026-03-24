import torch
from model import SymbolDecoder

PAD = 4


model = SymbolDecoder(vocab_size=5, embed_dim=16, hidden_dim=32, num_classes=4)
model.load_state_dict(torch.load("model.pth"))
model.eval()


def encode_signal(signal):
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    coded = [mapping[ch] for ch in signal]
    padded = coded + [PAD] * (7 - len(coded))
    return padded

def predict(signal):
    encoded = encode_signal(signal)
    inp = torch.tensor([encoded], dtype=torch.long)
    logits = model(inp)
    pred_class = torch.argmax(logits, dim=1).item()

    mapping = {0: "ATTACK", 1: "DEFEND", 2: "RETREAT", 3: "HOLD"}
    return mapping[pred_class]

def true_answer(signal):
    a_count = signal.count("A")
    b_count = signal.count("B")
    c_count = signal.count("C")
    d_count = signal.count("D")

    if signal[0] == "A":
        return "ATTACK"
    elif signal[-1] == "D":
        return "DEFEND"
    elif c_count > a_count and c_count > b_count and c_count > d_count:
        return "RETREAT"
    else:
        return "HOLD"

signal = input("Enter the secret message (e.g., ABCD): ").strip().upper()

print(f"AI Prediction: {predict(signal)}")
print(f"True Answer:  {true_answer(signal)}")
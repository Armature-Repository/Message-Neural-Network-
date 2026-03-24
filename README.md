Secret Symbol Language Decoder (Neural Network)

This project explores whether a neural network can learn to decode a secret symbolic language—a language whose rules are invented by the creator and never explicitly shown to the model. Inspired by the idea behind Mark Rober’s baseball sign decoder (but without video), this project focuses on pure sequence learning, pattern recognition, and rule inference.
The result is a compact but surprisingly powerful LSTM‑based model that learns to interpret symbolic sequences and classify them into meaningful actions.


Overview
You define a symbolic alphabet:
A, B, C, D

You define hidden rules that determine the meaning of a sequence. For example:
- If the sequence starts with A → ATTACK
- If the sequence ends with D → DEFEND
- If C is the most frequent symbol → RETREAT
- Otherwise → HOLD

You then generate thousands of sequences using these rules.
The neural network is trained only on the sequences and their labels — never the rules themselves.
The model’s job is to discover the rules on its own.


Key Features
- Custom symbolic language with hidden rule hierarchy
- Automatic dataset generation
- LSTM‑based sequence classifier
- Padding and embedding for variable‑length sequences
- Full training pipeline with accuracy and loss visualization
- Interactive prediction interface
- Modular project structure with a unified main.py controller


Project Structure
model.py             # LSTM model architecture
data_generator.py    # Creates symbolic sequences + labels
train.py             # Trains the neural network
predict.py           # Interactive decoder
main.py              # Unified interface for generate/train/predict
data.txt             # Generated dataset
loss.png             # Training loss curve
accuracy.png         # Training accuracy curve


How It Works
1. Generate Data
A dataset of symbolic sequences is created using the hidden rule system.
Each sequence is encoded numerically and padded with a dedicated PAD token.
2. Train the Model
An LSTM reads each sequence and learns to classify it into one of four meanings:
0 = ATTACK
1 = DEFEND
2 = RETREAT
3 = HOLD


Training produces:
- model.pth — the trained weights
- loss.png — loss curve
- accuracy.png — accuracy curve
3. Predict
You enter a sequence like:
ABC


The model outputs:
ATTACK


You can also compare it to the true rule‑based answer.

Example
Input:
C C A


Model prediction:
RETREAT


True rule‑based meaning:
RETREAT


Why This Project Stands Out
This project demonstrates:
- Sequence modeling
- Embeddings
- LSTM architectures
- Rule inference
- Data generation
- Model evaluation
- Interactive ML systems
It’s a compact but conceptually rich example of how neural networks can learn structure and hierarchy from symbolic data — a core idea in natural language processing, cryptography, and pattern recognition.
It’s also extremely easy to demo:
Type a sequence → watch the AI decode your secret language.


Future Extensions
- Figure out the rule from what the network knows
- Add noise or corrupted symbols
- Add multiple “dialects” with different rule sets
- Add hierarchical meanings (e.g., ATTACK + FAST)
- Add a GUI or web interface
- Replace LSTM with a Transformer
- Expand the symbol alphabet
- Add multi‑label outputs

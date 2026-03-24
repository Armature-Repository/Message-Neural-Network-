import torch
import torch.nn as nn

class SymbolDecoder(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=16, hidden_dim=32, num_classes=4):
        super().__init__()

        # Convert integers (0,1,2,3) into learned vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM to process the sequence
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len)
        """

        # Convert ints → embeddings
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Run through LSTM
        output, (h_n, c_n) = self.lstm(x)

        # h_n is the final hidden state → use it for classification
        logits = self.fc(h_n[-1])  # (batch, num_classes)

        return logits
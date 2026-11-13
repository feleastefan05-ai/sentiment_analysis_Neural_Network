import torch
import torch.nn as nn

from data import MAX_LEN, PAD, VOCAB_SIZE


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))

        # lungimea reala a fiecarei recenzii (fara <pad>)
        lengths = (x != PAD).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )

        _, (h, _) = self.lstm(packed)
        # h[-2] = memoria finala citind inainte, h[-1] = citind inapoi
        h = torch.cat((h[-2], h[-1]), dim=1)

        return self.fc(self.dropout(h))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentimentLSTM().to(device)
    print(model)

    n_params = sum(p.numel() for p in model.parameters())
    print("Parametri:", f"{n_params:,}")

    x = torch.randint(2, VOCAB_SIZE, (64, MAX_LEN)).to(device)
    out = model(x)
    print("Forma iesirii:", out.shape)
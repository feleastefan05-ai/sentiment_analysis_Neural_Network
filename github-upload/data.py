import re
from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split

VOCAB_SIZE = 20000
MAX_LEN = 256
PAD, UNK = 0, 1


def tokenize(text):
    text = text.lower().replace("<br />", " ")
    return re.findall(r"[a-z']+", text)


def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    ids = [vocab.get(w, UNK) for w in tokenize(text)]
    ids = ids[:MAX_LEN]                      # taie ce e prea lung
    ids += [PAD] * (MAX_LEN - len(ids))      # completeaza ce e prea scurt
    return ids


class ReviewDataset(Dataset):
    def __init__(self, split, vocab):
        self.x = torch.tensor([encode(t, vocab) for t in split["text"]])
        self.y = torch.tensor(list(split["label"]))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_dataloaders(batch_size=64):
    data = load_dataset("stanfordnlp/imdb")
    vocab = build_vocab(data["train"]["text"])

    full_train = ReviewDataset(data["train"], vocab)
    test = ReviewDataset(data["test"], vocab)

    gen = torch.Generator().manual_seed(42)
    train, val = random_split(full_train, [22500, 2500], generator=gen)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        vocab,
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = get_dataloaders()
    x, y = next(iter(train_loader))
    print("Forma pachetului:", x.shape, y.shape)
    print("Primele 20 numere din prima recenzie:", x[0][:20].tolist())
    print("Etichete din primul pachet:", y[:16].tolist())
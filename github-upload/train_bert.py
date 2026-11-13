import json
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH = 16
EPOCHS = 3
LR = 2e-5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Folosesc:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data = load_dataset("stanfordnlp/imdb")


def prepare(split):
    texts = [t.replace("<br />", " ") for t in split["text"]]
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN,
                    padding="max_length", return_tensors="pt")
    labels = torch.tensor(split["label"])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)


print("Transform textele in numere (dureaza ~1 minut)...")
full_train = prepare(data["train"])
test = prepare(data["test"])

# aceeasi impartire ca la LSTM (acelasi seed), ca sa comparam corect
gen = torch.Generator().manual_seed(42)
train, val = random_split(full_train, [22500, 2500], generator=gen)

train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val, batch_size=64)
test_loader = DataLoader(test, batch_size=64)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)
    return correct / total


history = {"train_loss": [], "val_acc": []}
best_val = 0.0
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for step, (ids, mask, y) in enumerate(train_loader, 1):
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        loss = loss_fn(logits.float(), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        if step % 200 == 0:
            print(f"  epoca {epoch}, pas {step}/{len(train_loader)}, "
                  f"loss: {loss.item():.4f}")

    train_loss = total_loss / len(train_loader.dataset)
    val_acc = evaluate(val_loader)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoca {epoch}/{EPOCHS}  loss: {train_loss:.4f}  "
          f"acuratete validare: {val_acc:.2%}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "bert_best.pt")

train_time = time.time() - t0

model.load_state_dict(torch.load("bert_best.pt"))
test_acc = evaluate(test_loader)
print(f"\nAcuratete finala pe test: {test_acc:.2%}")
print(f"Timp de antrenare: {train_time:.0f} secunde")

history["test_acc"] = test_acc
history["train_time"] = train_time
with open("bert_history.json", "w") as f:
    json.dump(history, f)
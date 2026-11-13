import json
import time

import torch
import torch.nn as nn

from data import get_dataloaders
from model import SentimentLSTM

EPOCHS = 30
PATIENCE = 4
LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Folosesc:", device)

train_loader, val_loader, test_loader, vocab = get_dataloaders()
model = SentimentLSTM().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def evaluate(loader):
    model.eval()                      # opreste dropout-ul
    correct, total = 0, 0
    with torch.no_grad():             # nu calculam gradienti, doar testam
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


history = {"train_loss": [], "val_acc": []}
best_val = 0.0
bad_epochs = 0
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()                     # porneste dropout-ul
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()         # sterge gradientii vechi
        loss = loss_fn(model(x), y)   # ghiceste + calculeaza loss-ul
        loss.backward()               # calculeaza directia de corectie
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # limiteaza pasii prea mari
        optimizer.step()              # mica parametrii

        total_loss += loss.item() * len(y)

    train_loss = total_loss / len(train_loader.dataset)
    val_acc = evaluate(val_loader)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoca {epoch}/{EPOCHS}  loss: {train_loss:.4f}  "
          f"acuratete validare: {val_acc:.2%}")

    if val_acc > best_val:            # pastram cel mai bun model
        best_val = val_acc
        bad_epochs = 0
        torch.save({"model": model.state_dict(), "vocab": vocab}, "lstm_best.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("Validarea nu mai creste, ma opresc.")
            break

train_time = time.time() - t0

# testul final, pe recenziile pe care reteaua nu le-a vazut niciodata
checkpoint = torch.load("lstm_best.pt")
model.load_state_dict(checkpoint["model"])
test_acc = evaluate(test_loader)
print(f"\nAcuratete finala pe test: {test_acc:.2%}")
print(f"Timp de antrenare: {train_time:.0f} secunde")

history["test_acc"] = test_acc
history["train_time"] = train_time
with open("lstm_history.json", "w") as f:
    json.dump(history, f)
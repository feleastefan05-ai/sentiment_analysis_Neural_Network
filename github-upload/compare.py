import json

import matplotlib.pyplot as plt

with open("lstm_history.json") as f:
    lstm = json.load(f)
with open("bert_history.json") as f:
    bert = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# stanga: acuratetea de validare pe epoci
ax1.plot(range(1, len(lstm["val_acc"]) + 1),
         [a * 100 for a in lstm["val_acc"]], label="LSTM", marker="o", ms=3)
ax1.plot(range(1, len(bert["val_acc"]) + 1),
         [a * 100 for a in bert["val_acc"]], label="DistilBERT", marker="o")
ax1.set_xlabel("Epoca")
ax1.set_ylabel("Acuratete validare (%)")
ax1.set_title("Invatarea pe epoci")
ax1.legend()
ax1.grid(alpha=0.3)

# dreapta: scorul final pe test
names = ["LSTM", "DistilBERT"]
scores = [lstm["test_acc"] * 100, bert["test_acc"] * 100]
bars = ax2.bar(names, scores, color=["#4c72b0", "#dd8452"])
ax2.set_ylim(80, 95)
ax2.set_ylabel("Acuratete test (%)")
ax2.set_title("Scor final pe test")
for bar, s in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, s + 0.2,
             f"{s:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("Salvat in results.png")
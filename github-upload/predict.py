import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import encode
from model import SentimentLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"

# --- incarcam LSTM-ul ---
ckpt = torch.load("lstm_best.pt")
vocab = ckpt["vocab"]
lstm = SentimentLSTM().to(device)
lstm.load_state_dict(ckpt["model"])
lstm.eval()

# --- incarcam DistilBERT ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(device)
bert.load_state_dict(torch.load("bert_best.pt"))
bert.eval()


def predict_lstm(text):
    x = torch.tensor([encode(text, vocab)]).to(device)
    with torch.no_grad():
        return torch.softmax(lstm(x), dim=1)[0]


def predict_bert(text):
    text = text.replace("<br />", " ")
    enc = tokenizer(text, truncation=True, max_length=256,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        return torch.softmax(bert(**enc).logits, dim=1)[0]


def show(name, probs):
    label = "POZITIV" if probs[1] > probs[0] else "NEGATIV"
    print(f"  {name:<11} {label}  ({probs.max().item():.0%})")


def compare(text):
    print(f"\n> {text}")
    show("LSTM", predict_lstm(text))
    show("DistilBERT", predict_bert(text))


examples = [
    "This movie was absolutely wonderful, I loved every minute of it!",
    "Terrible film. Boring plot and awful acting.",
    "This movie was not good at all.",
    "I expected to hate it, but it turned out to be amazing.",
    "Oh great, another boring sequel. Just what the world needed.",
]
for e in examples:
    compare(e)

print("\n--- Acum incearca tu (scrie 'q' ca sa iesi) ---")
while True:
    text = input("\nRecenzie: ").strip()
    if text.lower() == "q":
        break
    if text:
        compare(text)
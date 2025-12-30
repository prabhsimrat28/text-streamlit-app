import pandas as pd
import re
import random
import time
import matplotlib.pyplot as plt
from collections import Counter
import spacy
from spacy.util import minibatch, fix_random_seed
from spacy.training.example import Example
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

spam = pd.read_csv("spam.csv")

spam["length"] = spam["text"].apply(len)
spam["length"].hist(bins=50)
plt.show()

words = " ".join(spam["text"]).split()
print(Counter(words).most_common(10))
print(spam.sample(5))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

spam["text"] = spam["text"].apply(clean_text)

nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat")

textcat.add_label("ham")
textcat.add_label("spam")

train_texts = spam["text"].values
train_labels = [
    {"cats": {"ham": label == "ham", "spam": label == "spam"}}
    for label in spam["label"]
]

train_data = list(zip(train_texts, train_labels))

random.seed(1)
fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    batches = minibatch(train_data, size=8)
    for batch in batches:
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(losses)

docs = [nlp(text) for text in spam["text"]]
true_labels = spam["label"].values
preds = [max(doc.cats, key=doc.cats.get) for doc in docs]

print("Accuracy:", accuracy_score(true_labels, preds))
print("Precision:", precision_score(true_labels, preds, pos_label="spam"))
print("Recall:", recall_score(true_labels, preds, pos_label="spam"))
print("F1 Score:", f1_score(true_labels, preds, pos_label="spam"))

live_inputs = [
    "Congratulations you won a free ticket",
    "Can we meet tomorrow?",
    "URGENT!!! Claim your prize now"
]

for msg in live_inputs:
    doc = nlp(msg)
    print(msg, "→", max(doc.cats, key=doc.cats.get))
    time.sleep(1)

experiments = [
    "FREE FREE FREE",
    "free free free",
    "Free 🎉🎉",
    "F R E E",
    "This is definitely not spam"
]

for text in experiments:
    doc = nlp(text)
    print(text, "→", max(doc.cats, key=doc.cats.get))

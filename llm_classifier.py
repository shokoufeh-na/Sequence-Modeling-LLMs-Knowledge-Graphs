import pandas as pd
import re
from collections import Counter
import torch
import torch.nn as nn

# Load news dataset
news = pd.read_csv("news.csv")

# Select top 10 companies
top10 = (
    news.groupby("symbol")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

# Keep headlines from top 10 companies
df = news[news["symbol"].isin(top10.index)].copy()


# =====================================
# 1. Text preprocessing and tokenization
# =====================================

def tokenize(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Split into individual words
    tokens = text.split()

    return tokens


# Example
example = df["headline"].iloc[0]

print("Original:", example)
print("Tokens:", tokenize(example))


# =====================================
# 2. Build vocabulary
# =============================
# ========

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_vocab(headlines):

    word_counts = Counter()

    for headline in headlines:
        word_counts.update(tokenize(headline))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word in word_counts:
        vocab[word] = len(vocab)

    return vocab


vocab = build_vocab(df["headline"])

print("\nVocabulary size:", len(vocab))
print("Vocabulary example:", list(vocab.items())[:15])


# =====================================
# 3. Convert tokens into numerical IDs
# =====================================

def encode_text(text, vocab):

    tokens = tokenize(text)

    token_ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]

    return token_ids


example_ids = encode_text(example, vocab)

print("\nToken IDs:", example_ids)



class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_classes=7
    ):
        super().__init__()

        # 1. Token embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # 2. LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # 3. Classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids):

        # Convert token IDs into embeddings
        embedded = self.embedding(token_ids)

        # Process embeddings through LSTM
        output, (hidden, cell) = self.lstm(embedded)

        # Get final hidden state
        final_hidden = hidden[-1]

        # Predict 7 market-impact classes
        logits = self.fc(final_hidden)

        return logits



 
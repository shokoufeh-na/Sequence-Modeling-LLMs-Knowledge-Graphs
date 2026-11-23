# import pandas as pd
# import re
# from collections import Counter
# import torch.nn as nn

# # Load news dataset
# news = pd.read_csv("news.csv")

# # Select top 10 companies
# top10 = (
#     news.groupby("symbol")
#     .size()
#     .sort_values(ascending=False)
#     .head(10)
# )

# # Keep headlines from top 10 companies
# df = news[news["symbol"].isin(top10.index)].copy()


# # =====================================
# # 1. Text preprocessing and tokenization
# # =====================================

# def tokenize(text):

#     # Convert to lowercase
#     text = str(text).lower()

#     # Remove punctuation
#     text = re.sub(r"[^a-z0-9\s]", " ", text)

#     # Split into individual words
#     tokens = text.split()

#     return tokens


# # Example
# example = df["headline"].iloc[0]

# print("Original:", example)
# print("Tokens:", tokenize(example))


# # =====================================
# # 2. Build vocabulary
# # =============================
# # ========

# PAD_TOKEN = "<PAD>"
# UNK_TOKEN = "<UNK>"

# def build_vocab(headlines):

#     word_counts = Counter()

#     for headline in headlines:
#         word_counts.update(tokenize(headline))

#     vocab = {
#         PAD_TOKEN: 0,
#         UNK_TOKEN: 1
#     }

#     for word in word_counts:
#         vocab[word] = len(vocab)

#     return vocab


# vocab = build_vocab(df["headline"])

# print("\nVocabulary size:", len(vocab))
# print("Vocabulary example:", list(vocab.items())[:15])


# # =====================================
# # 3. Convert tokens into numerical IDs
# # =====================================

# def encode_text(text, vocab):

#     tokens = tokenize(text)

#     token_ids = [
#         vocab.get(token, vocab[UNK_TOKEN])
#         for token in tokens
#     ]

#     return token_ids


# example_ids = encode_text(example, vocab)

# print("\nToken IDs:", example_ids)


# class LSTMClassifier(nn.Module):

#     def __init__(
#         self,
#         vocab_size,
#         embedding_dim=100,
#         hidden_dim=128,
#         num_layers=2,
#         num_classes=7
#     ):
#         super().__init__()

#         # 1. Token embedding
#         self.embedding = nn.Embedding(
#             vocab_size,
#             embedding_dim,
#             padding_idx=0
#         )

#         # 2. LSTM layers
#         self.lstm = nn.LSTM(
#             input_size=embedding_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=0.2
#         )

#         # 3. Classification layer
#         self.fc = nn.Linear(hidden_dim, num_classes)

#     def forward(self, token_ids):

#         # Convert token IDs into embeddings
#         embedded = self.embedding(token_ids)

#         # Process embeddings through LSTM
#         output, (hidden, cell) = self.lstm(embedded)

#         # Get final hidden state
#         final_hidden = hidden[-1]

#         # Predict 7 market-impact classes
#         logits = self.fc(final_hidden)

#         return logits



import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 1. Data Loading, Top-10 Filtering & Chronological Split
# ---------------------------------------------------------
def prepare_data(news_path: str, impacts_path: str):
    news_df = pd.read_csv(news_path)
    impacts_df = pd.read_csv(impacts_path)

    # Identify top 10 companies by headline count
    top_10_symbols = news_df["symbol"].value_counts().nlargest(10).index.tolist()
    news_df = news_df[news_df["symbol"].isin(top_10_symbols)].copy()

    # Standardize dates to YYYY-MM-DD for accurate matching
    news_df["date"] = pd.to_datetime(news_df["date"], utc=True).dt.strftime("%Y-%m-%d")
    impacts_df["date"] = pd.to_datetime(impacts_df["date"], utc=True).dt.strftime("%Y-%m-%d")

    # Combine multiple headlines for the same company on the same date
    grouped_news = (
        news_df.groupby(["symbol", "date"], as_index=False)
        .agg({"headline": lambda x: " . ".join(x.dropna().astype(str))})
    )

    # Merge with historical impacts (assumes 'impact_score' column exists in [-3..3])
    merged_df = pd.merge(grouped_news, impacts_df, on=["symbol", "date"], how="inner")
    merged_df = merged_df.dropna(subset=["headline", "impact_score"]).copy()

    # Map labels [-3, -2, -1, 0, 1, 2, 3] -> [0, 1, 2, 3, 4, 5, 6]
    merged_df["label"] = merged_df["impact_score"].astype(int) + 3

    # Chronological split based on unique sorted dates (70% / 15% / 15%)
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(merged_df["date"].unique())
    train_cutoff = unique_dates[int(0.70 * len(unique_dates))]
    val_cutoff = unique_dates[int(0.85 * len(unique_dates))]

    train_df = merged_df[merged_df["date"] <= train_cutoff].reset_index(drop=True)
    val_df = merged_df[(merged_df["date"] > train_cutoff) & (merged_df["date"] <= val_cutoff)].reset_index(drop=True)
    test_df = merged_df[merged_df["date"] > val_cutoff].reset_index(drop=True)

    return train_df, val_df, test_df, top_10_symbols


# ---------------------------------------------------------
# 2. Text Preprocessing, Vocabulary & Dataset
# ---------------------------------------------------------
def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\-\$\%]", " ", text)
    return text.split()


class Vocabulary:
    def __init__(self, max_vocab_size: int = 20000, min_freq: int = 2):
        self.pad_idx = 0
        self.unk_idx = 1
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

    def build(self, texts: list[str]):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for word, freq in counter.most_common(self.max_vocab_size - 2):
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)

    def encode(self, text: str, max_len: int = 128) -> tuple[list[int], int]:
        tokens = tokenize(text)[:max_len]
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        length = max(len(ids), 1)
        if len(ids) < max_len:
            ids += [self.pad_idx] * (max_len - len(ids))
        return ids, length


class FinancialNewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = 128):
        self.sequences = []
        self.lengths = []
        self.labels = df["label"].values

        for text in df["headline"]:
            ids, length = vocab.encode(text, max_len=max_len)
            self.sequences.append(ids)
            self.lengths.append(length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------
# 3. PyTorch LSTM Classification Model
# ---------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.emb_dropout(self.embedding(input_ids))

        # Pack padded sequence so PAD tokens do not distort hidden states
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # Concatenate final forward and backward hidden states from top layer
        if self.lstm.bidirectional:
            feat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            feat = hidden[-1]

        return self.fc(feat)


# ---------------------------------------------------------
# 4. Training & Evaluation Loops
# ---------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for seqs, lengths, labels in loader:
            seqs, lengths, labels = seqs.to(device), lengths.to(device), labels.to(device)
            logits = model(seqs, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


def train_model(
    news_path="news.csv",
    impacts_path="historical_impacts_values.csv",
    epochs=10,
    batch_size=64,
    lr=1e-3,
    max_len=128,
):
    train_df, val_df, test_df, top_10 = prepare_data(news_path, impacts_path)
    print(f"Top 10 companies: {top_10}")
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Fit vocabulary ONLY on training text to avoid data leakage
    vocab = Vocabulary(max_vocab_size=20000, min_freq=2)
    vocab.build(train_df["headline"].tolist())

    train_loader = DataLoader(FinancialNewsDataset(train_df, vocab, max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FinancialNewsDataset(val_df, vocab, max_len), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FinancialNewsDataset(test_df, vocab, max_len), batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(vocab_size=len(vocab.word2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, lengths, labels in train_loader:
            seqs, lengths, labels = seqs.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_lstm_model.pt")

    # Final evaluation on test set (shift labels back to [-3..3])
    model.load_state_dict(torch.load("best_lstm_model.pt"))
    test_loss, test_f1, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Macro-F1: {test_f1:.4f}")
    print(classification_report(labels - 3, preds - 3, digits=4, zero_division=0))


if __name__ == "__main__":
    train_model()

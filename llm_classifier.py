# import pandas as pd
# import re
# from collections import Counter
# import torch.nn as nn
# import pandas as pd

# news_df = pd.read_csv("news.csv")

# # 1. Count headlines per row and find top 10 tickers
# news_df["headline_count"] = news_df["headline"].astype(str).str.count("\x1f") + 1

# top_10_symbols = (
#     news_df.groupby("symbol")["headline_count"]
#     .sum()
#     .sort_values(ascending=False)
#     .head(10)
#     .index.tolist()  # <-- Convert index to a plain Python list of ticker strings!
# )
# # print("Top 10 tickers:", top_10_symbols)

# # 2. Filter using the list of ticker strings
# news_df = news_df[news_df["symbol"].isin(top_10_symbols)].copy()
# # print("Rows after top-10 filter:", len(news_df))

# # 3. Split on '\x1f'
# news_df["date"] = news_df["date"].astype(str).str.split("\x1f")
# news_df["headline"] = news_df["headline"].astype(str).str.split("\x1f")

# ##########

# valid_mask = news_df["date"].str.len() == news_df["headline"].str.len()
# news_df = news_df[valid_mask].explode(["date", "headline"], ignore_index=True)

# # 2. Slice Eastern-time date to YYYY-MM-DD and clean whitespace
# news_df["date"] = news_df["date"].str.strip().str.slice(0, 10)
# news_df["headline"] = news_df["headline"].str.strip()

# print(news_df["date"].head(10))
# print(news_df["headline"].head(10))

# # 3. Combine same-day headlines with ' . ' so row 2061 becomes clean text
# company_day_df = (
#     news_df.groupby(["symbol", "date"], as_index=False)
#     .agg({"headline": lambda x: " . ".join(x.dropna().astype(str))})
# )


# Load news dataset
# news = pd.read_csv("historical_impacts_values-latest.csv")
# # 1. Read the combined CSV file
# print(news.head())
# print(news.info())
# print(news.isnull().sum())
# print(news.iloc[206])
# print(news["symbol"].unique().tolist())
# row = news_df.loc[6]  # Or use news_df[news_df["symbol"] == "EPS"].iloc[0]

# for col, val in row.items():
#     print(f"=== {col} ===")
#     print(repr(val))
#     print()

# news_df['headline_count'] = news_df['headline'].str.count('\x1f') + 1
 
# # Identify top 10 companies by headline count
# top_10_symbols = (
#     news_df.groupby('symbol')['headline_count']
# .sum()
# .sort_values(ascending=False)
#     .head(10)
# )
# news_df = news_df[news_df["symbol"].isin(top_10_symbols)].copy()
# print(top_10_symbols.head(10))
 

# # 1. Split the '\x1f'-delimited strings into lists
# news_df["date"] = news_df["date"].astype(str).str.split("\x1f")
# news_df["headline"] = news_df["headline"].astype(str).str.split("\x1f")

# print(news_df["date"].head(10))

# # 2. Count individual headlines per row: (number of \x1f separators) + 1
# news_df['headline_count'] = news_df['headline'].str.count('\x1f') + 1

# # 3. Group by company symbol, sum the headline counts, and take the top 10
# top_10_companies = (
#     news_df.groupby('symbol')['headline_count']
#     .sum()
#     .sort_values(ascending=False)
#     .head(10)
# )

# # print(top_10_companies)
# # news_df = news_df[news_df["symbol"].isin(top_10_companies)].copy()
# print(news_df.head(10))

# 1. Split the '\x1f'-delimited strings into lists
# news_df["date"] = news_df["date"].astype(str).str.split("\x1f")
# news_df["headline"] = news_df["headline"].astype(str).str.split("\x1f")
# print(news_df.head(10))

# # 2. Verify each row has the same number of dates and headlines
# valid_mask = news_df["date"].str.len() == news_df["headline"].str.len()
# news_df = news_df[valid_mask].copy()

# # 3. Explode both columns together so each (date, headline) pair becomes its own row
# news_df = news_df.explode(["date", "headline"], ignore_index=True)

# # 4. Standardize each individual Eastern-time timestamp to YYYY-MM-DD
# news_df["date"] = news_df["date"].str.strip().str.slice(0, 10)
# news_df["headline"] = news_df["headline"].str.strip()

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



# import re
# from collections import Counter
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import classification_report, f1_score

# # Set seeds for reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---------------------------------------------------------
# # 1. Data Loading, Top-10 Filtering & Chronological Split
# # ---------------------------------------------------------
# def prepare_data(news_path: str, impacts_path: str):
#     news_df = pd.read_csv(news_path)
#     news_df = pd.read_csv(news_path)
#     impacts_df = pd.read_csv(impacts_path)

#     # 1. Count headlines per row and find top 10 tickers
#     news_df["headline_count"] = news_df["headline"].astype(str).str.count("\x1f") + 1

#     top_10_symbols = (
#         news_df.groupby("symbol")["headline_count"]
#         .sum()
#         .sort_values(ascending=False)
#         .head(10)
#         .index.tolist()  # <-- Convert index to a plain Python list of ticker strings!
#     )
#     # print("Top 10 tickers:", top_10_symbols)

#     # 2. Filter using the list of ticker strings
#     news_df = news_df[news_df["symbol"].isin(top_10_symbols)].copy()
#     # print("Rows after top-10 filter:", len(news_df))

#     # 3. Split on '\x1f'
#     news_df["date"] = news_df["date"].astype(str).str.split("\x1f")
#     news_df["headline"] = news_df["headline"].astype(str).str.split("\x1f")

#     valid_mask = news_df["date"].str.len() == news_df["headline"].str.len()
#     news_df = news_df[valid_mask].explode(["date", "headline"], ignore_index=True)

#     # 2. Slice Eastern-time date to YYYY-MM-DD and clean whitespace
#     news_df["date"] = news_df["date"].str.strip().str.slice(0, 10)
#     news_df["headline"] = news_df["headline"].str.strip()

#     print(news_df["date"].head(10))
#     print(news_df["headline"].head(10))

#     # Combine multiple headlines for the same company on the same date
#     grouped_news = (
#         news_df.groupby(["symbol", "date"], as_index=False)
#         .agg({"headline": lambda x: " . ".join(x.dropna().astype(str))})
#     )
#     #####
#     # Merge with historical impacts (assumes 'impact_score' column exists in [-3..3])
#     merged_df = pd.merge(grouped_news, impacts_df, on=["symbol", "date"], how="inner")
#     merged_df = merged_df.dropna(subset=["headline", "impact_score"]).copy()

#     # Map labels [-3, -2, -1, 0, 1, 2, 3] -> [0, 1, 2, 3, 4, 5, 6]
#     merged_df["label"] = merged_df["impact_score"].astype(int) + 3

#     # Chronological split based on unique sorted dates (70% / 15% / 15%)
#     merged_df = merged_df.sort_values("date").reset_index(drop=True)
#     unique_dates = sorted(merged_df["date"].unique())
#     train_cutoff = unique_dates[int(0.70 * len(unique_dates))]
#     val_cutoff = unique_dates[int(0.85 * len(unique_dates))]

#     train_df = merged_df[merged_df["date"] <= train_cutoff].reset_index(drop=True)
#     val_df = merged_df[(merged_df["date"] > train_cutoff) & (merged_df["date"] <= val_cutoff)].reset_index(drop=True)
#     test_df = merged_df[merged_df["date"] > val_cutoff].reset_index(drop=True)

#     return train_df, val_df, test_df, top_10_symbols


# # ---------------------------------------------------------
# # 2. Text Preprocessing, Vocabulary & Dataset
# # ---------------------------------------------------------
# def tokenize(text: str) -> list[str]:
#     text = text.lower()
#     text = re.sub(r"[^a-z0-9\s\.\-\$\%]", " ", text)
#     return text.split()


# class Vocabulary:
#     def __init__(self, max_vocab_size: int = 20000, min_freq: int = 2):
#         self.pad_idx = 0
#         self.unk_idx = 1
#         self.word2idx = {"<PAD>": 0, "<UNK>": 1}
#         self.max_vocab_size = max_vocab_size
#         self.min_freq = min_freq

#     def build(self, texts: list[str]):
#         counter = Counter()
#         for text in texts:
#             counter.update(tokenize(text))
#         for word, freq in counter.most_common(self.max_vocab_size - 2):
#             if freq >= self.min_freq:
#                 self.word2idx[word] = len(self.word2idx)

#     def encode(self, text: str, max_len: int = 128) -> tuple[list[int], int]:
#         tokens = tokenize(text)[:max_len]
#         ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
#         length = max(len(ids), 1)
#         if len(ids) < max_len:
#             ids += [self.pad_idx] * (max_len - len(ids))
#         return ids, length


# class FinancialNewsDataset(Dataset):
#     def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = 128):
#         self.sequences = []
#         self.lengths = []
#         self.labels = df["label"].values

#         for text in df["headline"]:
#             ids, length = vocab.encode(text, max_len=max_len)
#             self.sequences.append(ids)
#             self.lengths.append(length)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.sequences[idx], dtype=torch.long),
#             torch.tensor(self.lengths[idx], dtype=torch.long),
#             torch.tensor(self.labels[idx], dtype=torch.long),
#         )


# # ---------------------------------------------------------
# # 3. PyTorch LSTM Classification Model
# # ---------------------------------------------------------
# class LSTMClassifier(nn.Module):
#     def __init__(
#         self,
#         vocab_size: int,
#         embed_dim: int = 128,
#         hidden_dim: int = 128,
#         num_layers: int = 2,
#         num_classes: int = 7,
#         dropout: float = 0.3,
#         bidirectional: bool = True,
#     ):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.emb_dropout = nn.Dropout(dropout)

#         self.lstm = nn.LSTM(
#             input_size=embed_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0,
#             bidirectional=bidirectional,
#         )

#         lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
#         self.fc = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(lstm_out_dim, num_classes),
#         )

#     def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
#         embedded = self.emb_dropout(self.embedding(input_ids))

#         # Pack padded sequence so PAD tokens do not distort hidden states
#         packed = nn.utils.rnn.pack_padded_sequence(
#             embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
#         )
#         _, (hidden, _) = self.lstm(packed)

#         # Concatenate final forward and backward hidden states from top layer
#         if self.lstm.bidirectional:
#             feat = torch.cat((hidden[-2], hidden[-1]), dim=1)
#         else:
#             feat = hidden[-1]

#         return self.fc(feat)


# # ---------------------------------------------------------
# # 4. Training & Evaluation Loops
# # ---------------------------------------------------------
# def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
#     model.eval()
#     total_loss = 0.0
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for seqs, lengths, labels in loader:
#             seqs, lengths, labels = seqs.to(device), lengths.to(device), labels.to(device)
#             logits = model(seqs, lengths)
#             loss = criterion(logits, labels)
#             total_loss += loss.item() * labels.size(0)

#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     avg_loss = total_loss / len(loader.dataset)
#     macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
#     return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


# def train_model(
#     news_path="news.csv",
#     impacts_path="historical_impacts_values-latest.csv",
#     epochs=10,
#     batch_size=64,
#     lr=1e-3,
#     max_len=128,
# ):
#     train_df, val_df, test_df, top_10 = prepare_data(news_path, impacts_path)
#     print(f"Top 10 companies: {top_10}")
#     print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

#     # Fit vocabulary ONLY on training text to avoid data leakage
#     vocab = Vocabulary(max_vocab_size=20000, min_freq=2)
#     vocab.build(train_df["headline"].tolist())

#     train_loader = DataLoader(FinancialNewsDataset(train_df, vocab, max_len), batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(FinancialNewsDataset(val_df, vocab, max_len), batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(FinancialNewsDataset(test_df, vocab, max_len), batch_size=batch_size, shuffle=False)

#     model = LSTMClassifier(vocab_size=len(vocab.word2idx)).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

#     best_val_f1 = -1.0
#     for epoch in range(1, epochs + 1):
#         model.train()
#         train_loss = 0.0
#         for seqs, lengths, labels in train_loader:
#             seqs, lengths, labels = seqs.to(device), lengths.to(device), labels.to(device)
#             optimizer.zero_grad()
#             logits = model(seqs, lengths)
#             loss = criterion(logits, labels)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             train_loss += loss.item() * labels.size(0)

#         train_loss /= len(train_loader.dataset)
#         val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
#         print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             torch.save(model.state_dict(), "best_lstm_model.pt")

#     # Final evaluation on test set (shift labels back to [-3..3])
#     model.load_state_dict(torch.load("best_lstm_model.pt"))
#     test_loss, test_f1, preds, labels = evaluate(model, test_loader, criterion)
#     print(f"\nTest Loss: {test_loss:.4f} | Test Macro-F1: {test_f1:.4f}")
#     print(classification_report(labels - 3, preds - 3, digits=4, zero_division=0))


# if __name__ == "__main__":
#     train_model()





import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score

# Prevent TensorFlow from allocating GPU memory needed by PyTorch
tf.config.set_visible_devices([], "GPU")

# Set seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 1. Data Loading, Top-10 Filtering & Chronological Split
# ---------------------------------------------------------
def prepare_data(news_path: str, impacts_path: str):
    news_df = pd.read_csv(news_path)
    impacts_df = pd.read_csv(impacts_path)

    # 1. Count headlines per row and find top 10 tickers
    news_df["headline_count"] = news_df["headline"].astype(str).str.count("\x1f") + 1

    top_10_symbols = (
        news_df.groupby("symbol")["headline_count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    # 2. Filter using the list of ticker strings
    news_df = news_df[news_df["symbol"].isin(top_10_symbols)].copy()

    # 3. Split on '\x1f'
    news_df["date"] = news_df["date"].astype(str).str.split("\x1f")
    news_df["headline"] = news_df["headline"].astype(str).str.split("\x1f")

    valid_mask = news_df["date"].str.len() == news_df["headline"].str.len()
    news_df = news_df[valid_mask].explode(["date", "headline"], ignore_index=True)

    # Slice Eastern-time date to YYYY-MM-DD and clean whitespace
    news_df["date"] = news_df["date"].str.strip().str.slice(0, 10)
    news_df["headline"] = news_df["headline"].str.strip()

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
# 2. TensorFlow / Keras Text Preprocessing, Vectorization & Dataset
# ---------------------------------------------------------
def custom_standardization(text_tensor: tf.Tensor) -> tf.Tensor:
    lowercase = tf.strings.lower(text_tensor)
    # Match original regex: keep alphanumeric, whitespace, ., -, $, %
    return tf.strings.regex_replace(lowercase, r"[^a-z0-9\s\.\-\$\%]", " ")


def build_vectorizer(
    texts: list[str],
    max_vocab_size: int = 20000,
    max_len: int = 128,
) -> tf.keras.layers.TextVectorization:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_vocab_size,
        standardize=custom_standardization,
        split="whitespace",
        output_mode="int",
        output_sequence_length=max_len,
    )
    vectorizer.adapt(texts)
    return vectorizer


class FinancialNewsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vectorizer: tf.keras.layers.TextVectorization,
    ):
        # Batch-vectorize all headlines at once into a NumPy array of shape (N, max_len)
        seqs_np = vectorizer(df["headline"].tolist()).numpy()
        self.sequences = torch.tensor(seqs_np, dtype=torch.long)

        # Compute unpadded sequence lengths (clamped to min=1 for pack_padded_sequence)
        self.lengths = (self.sequences != 0).sum(dim=1).clamp(min=1)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


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
    impacts_path="historical_impacts_values-latest.csv",
    epochs=10,
    batch_size=64,
    lr=1e-3,
    max_len=128,
    max_vocab_size=20000,
):
    train_df, val_df, test_df, top_10 = prepare_data(news_path, impacts_path)
    print(f"Top 10 companies: {top_10}")

    # Build and adapt Keras TextVectorization on training data only
    vectorizer = build_vectorizer(
        train_df["headline"].tolist(),
        max_vocab_size=max_vocab_size,
        max_len=max_len,
    )
    vocab_size = vectorizer.vocabulary_size()
    print(f"Vocabulary size: {vocab_size}")

    # Create PyTorch Datasets & DataLoaders
    train_dataset = FinancialNewsDataset(train_df, vectorizer)
    val_dataset = FinancialNewsDataset(val_df, vectorizer)
    test_dataset = FinancialNewsDataset(test_df, vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize PyTorch model, loss, and optimizer
    model = LSTMClassifier(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_lstm_model.pt")

    # Final evaluation on test set
    model.load_state_dict(torch.load("best_lstm_model.pt"))
    test_loss, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Macro-F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, zero_division=0))


if __name__ == "__main__":
    train_model()

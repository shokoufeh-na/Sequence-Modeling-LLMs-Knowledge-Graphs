import pandas as pd

# 1. Load news dataset
news = pd.read_csv("news.csv")

# 2. Find top 10 companies by headline count
top10 = (
    news.groupby("symbol")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 companies:")
print(top10)

# 3. Keep headlines from top 10 companies
top_symbols = top10.index.tolist()

df = news[news["symbol"].isin(top_symbols)].copy()

# 4. Keep headline text
df = df[["date", "symbol", "headline"]]

print("\nTotal headlines:", len(df))
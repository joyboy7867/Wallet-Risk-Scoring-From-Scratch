import pandas as pd
from fetch import fetch_erc20_transfers
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
df=pd.read_csv("dataset/Wallet id.csv")

wallets = df["wallet_id"].tolist()
all_data = []

for i, wallet in enumerate(wallets):
    print(f"Fetching wallet {i+1}/{len(wallets)}: {wallet}")
    txs = fetch_erc20_transfers(wallet)
    for tx in txs:
        tx['wallet'] = wallet
        all_data.append(tx)

df = pd.DataFrame(all_data)
from datetime import datetime
df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])

# Initialize feature list
features = []

# Group by wallet
for wallet, group in df.groupby("wallet"):
    total_tx = len(group)
    unique_tokens = group["token_symbol"].nunique()

    sent = group[group["from_address"].str.lower() == wallet.lower()]
    received = group[group["to_address"].str.lower() == wallet.lower()]

    total_sent = sent["value"].astype(float).sum()
    total_received = received["value"].astype(float).sum()
    avg_tx_value = group["value"].astype(float).mean()

    first_tx = group["block_timestamp"].min()
    last_tx = group["block_timestamp"].max()
    active_days = (last_tx - first_tx).days + 1

    features.append({
        "wallet": wallet,
        "num_transactions": total_tx,
        "unique_tokens": unique_tokens,
        "total_value_sent": total_sent,
        "total_value_received": total_received,
        "avg_tx_value": avg_tx_value,
        "first_tx": first_tx,
        "last_tx": last_tx,
        "active_days": active_days
    })

features_df = pd.DataFrame(features)
features_df["raw_risk_score"] = (
    (1 / (features_df["active_days"] + 1)) * 0.2 +
    (1 / (features_df["num_transactions"] + 1)) * 0.3 +
    (1 / (features_df["total_value_received"] + 1)) * 0.3 +
    (1 / (features_df["unique_tokens"] + 1)) * 0.2
)


features_df = features_df.replace([np.inf, -np.inf], np.nan)
features_df = features_df.dropna(subset=["raw_risk_score"])


features_df["risk_score"] = (
    (features_df["raw_risk_score"] - features_df["raw_risk_score"].min()) /
    (features_df["raw_risk_score"].max() - features_df["raw_risk_score"].min()) * 999 + 1
).astype(int)
features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
features_df.dropna(inplace=True)


numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns.drop("risk_score")



scaler = MinMaxScaler()
features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])

print("✅ Data cleaned and normalized.")
X = features_df.drop(columns=["wallet", "first_tx", "last_tx", "risk_score"])
y = features_df["risk_score"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")
model_predicted=model.predict(X)
features_df["model_predicted_score"]=model_predicted

features_df[["wallet","model_predicted"]].to_csv("wallet_score.csv")


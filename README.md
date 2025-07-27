# 🧠 Wallet Risk Scoring From Scratch

This project aims to compute a **risk score between 1 and 1000** for a list of Ethereum wallet addresses based on their on-chain behavior using the Compound V2/V3 protocol. The system fetches real-time transaction data via API, extracts meaningful features, and applies a simple weighted scoring model to rank wallet risk.

---

## 📦 Data Collection Method

- **Source**: Data is fetched via a third-party API (e.g., Covalent or Alchemy) using ERC-20 token transfer history endpoints.
- **Wallet Input**: A CSV file containing 100+ Ethereum wallet addresses.
- **Fetching Strategy**:
  - API requests are made sequentially for each wallet.
  - Each wallet's transaction history is parsed and stored in a unified DataFrame.
  - Only relevant fields like transaction timestamps, transfer amounts, and token types are retained.

---

## 🔍 Feature Selection Rationale

The following features were selected to infer behavioral patterns and financial activities of wallets:

| Feature Name          | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `active_days`         | Number of unique days the wallet was active (based on tx timestamps)       |
| `num_transactions`    | Total number of ERC-20 transactions performed or received                   |
| `total_value_received`| Cumulative value of incoming transfers (in token base units)               |
| `unique_tokens`       | Number of unique tokens received (diversity of interaction)                |

These features offer a balance of **activity frequency**, **volume**, and **diversification**, which are solid indicators for identifying high-risk or suspicious wallets.

---

## 📊 Scoring Method

The wallet risk score is calculated using a **heuristic inverse-weighted formula**, designed to give higher risk scores to wallets that are:
- **Less active**
- **Have fewer transactions**
- **Receive fewer funds**
- **Lack token diversity**

### Formula


features_df["risk_score"] = (
    (1 / (features_df["active_days"] + 1)) * 0.2 +
    (1 / (features_df["num_transactions"] + 1)) * 0.3 +
    (1 / (features_df["total_value_received"] + 1)) * 0.3 +
    (1 / (features_df["unique_tokens"] + 1)) * 0.2
)
The raw score (ranging from ~0 to ~1) is then scaled to a range of 1 to 1000 using:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1, 1000))
features_df["risk_score"] = scaler.fit_transform(features_df[["risk_score"]])

## ✅ Justification of Risk Indicators
Active Days: Less activity days may indicate inactive or sleeper wallets, which could be riskier.

Transaction Count: Wallets with fewer transactions are harder to analyze for intent or legitimacy.

Total Value Received: Receiving smaller or inconsistent amounts could indicate test or bot activity.

Unique Tokens: Limited interaction diversity may mean the wallet is tied to specific schemes or testing only a single dApp.

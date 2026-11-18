
import streamlit as st
import requests
import pandas as pd

# ---------------------------------
# 1. Page configuration
# ---------------------------------

st.set_page_config(
    page_title="FraudGuard AI - Monitoring",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FraudGuard AI")
st.subheader("Fraud Monitoring Dashboard")

API_URL = "http://127.0.0.1:8000/transactions"

# ---------------------------------
# 2. Fetch transaction history
# ---------------------------------

st.subheader("Transaction History")

if st.button("🔄 Refresh Data"):

    st.rerun()

try:

    response = requests.get(
        API_URL,
        timeout=15
    )

    response.raise_for_status()

    transactions = response.json()

    if not transactions:

        st.info(
            "No transaction records found. "
            "Please analyze a transaction first."
        )

    else:

        # Convert API response into DataFrame
        df = pd.DataFrame(transactions)

        # ---------------------------------
        # 3. Calculate dashboard metrics
        # ---------------------------------

        total_transactions = len(df)

        fraud_count = int(
            (df["prediction"] == 1).sum()
        )

        genuine_count = int(
            (df["prediction"] == 0).sum()
        )

        fraud_rate = (
            fraud_count / total_transactions
        ) * 100

        # ---------------------------------
        # 4. Display metrics
        # ---------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            total_transactions
        )

        col2.metric(
            "Potential Fraud",
            fraud_count
        )

        col3.metric(
            "Genuine Transactions",
            genuine_count
        )

        col4.metric(
            "Predicted Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

        st.divider()

        # ---------------------------------
        # 5. Transaction distribution chart
        # ---------------------------------

        st.subheader("Prediction Distribution")

        chart_data = pd.DataFrame({
            "Category": ["Potential Fraud", "Genuine"],
            "Count": [fraud_count, genuine_count]
        })

        st.bar_chart(
            chart_data.set_index("Category")
        )

        # ---------------------------------
        # 6. Fraud probability chart
        # ---------------------------------

        st.subheader("Fraud Probability Overview")

        probability_data = df[
            ["transaction_id", "fraud_probability"]
        ].set_index("transaction_id")

        st.line_chart(probability_data)

        # ---------------------------------
        # 7. Transaction history table
        # ---------------------------------

        st.subheader("All Analyzed Transactions")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # ---------------------------------
        # 8. Potential fraud records
        # ---------------------------------

        st.subheader("⚠️ Potential Fraud Alerts")

        fraud_records = df[
            df["prediction"] == 1
        ]

        if not fraud_records.empty:

            st.dataframe(
                fraud_records,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No potential fraud predictions "
                "in the current history."
            )

except requests.exceptions.ConnectionError:

    st.error(
        "Cannot connect to FastAPI. "
        "Please start the backend server."
    )

except requests.exceptions.RequestException as error:

    st.error(f"API request failed: {error}")

except (KeyError, ValueError) as error:

    st.error(f"Unexpected API response: {error}")

# ---------------------------------
# 9. Footer
# ---------------------------------

st.divider()

st.caption(
    "FraudGuard AI | AI/ML Fraud Monitoring System"
)


import streamlit as st
import requests

# --------------------------------
# 1. Page configuration
# --------------------------------

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------
# 2. Dashboard heading
# --------------------------------

st.title("🛡️ FraudGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "Enter transaction details to estimate "
    "the risk of potential credit card fraud."
)

st.divider()

# --------------------------------
# 3. API URL
# --------------------------------

API_URL = "http://127.0.0.1:8000/predict"

# --------------------------------
# 4. Transaction input form
# --------------------------------

st.header("Transaction Details")

with st.form("transaction_form"):

    time_value = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=10000.0
    )

    amount_value = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=150.0
    )

    st.subheader("Transaction Features")

    st.caption(
        "V1 to V28 are anonymized PCA-transformed "
        "features from the dataset."
    )

    transaction_features = {}

    col1, col2, col3, col4 = st.columns(4)

    for i in range(1, 29):

        with [col1, col2, col3, col4][(i - 1) % 4]:

            transaction_features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                step=0.1,
                key=f"feature_{i}"
            )

    submitted = st.form_submit_button(
        "🔍 Analyze Transaction",
        use_container_width=True
    )

# --------------------------------
# 5. Send transaction to FastAPI
# --------------------------------

if submitted:

    transaction_data = {
        "Time": time_value,
        **transaction_features,
        "Amount": amount_value
    }

    try:

        with st.spinner("Analyzing transaction..."):

            response = requests.post(
                API_URL,
                json=transaction_data,
                timeout=30
            )

        response.raise_for_status()

        result = response.json()

        # --------------------------------
        # 6. Display prediction
        # --------------------------------

        st.divider()
        st.header("Prediction Results")

        probability = result["fraud_probability"] * 100
        prediction = result["prediction"]

        col1, col2 = st.columns(2)

        col1.metric(
            "Fraud Risk Probability",
            f"{probability:.2f}%"
        )

        col2.metric(
            "Model Prediction",
            "Potential Fraud" if prediction == 1
            else "Genuine Transaction"
        )

        if prediction == 1:

            st.error(
                "⚠️ Potential Fraud Detected! "
                "This transaction requires further review."
            )

        else:

            st.success(
                "✅ Model classified this transaction "
                "as Genuine."
            )

        st.caption(
            "This is an ML risk estimate, not proof of fraud "
            "or a real bank authorization decision."
        )

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Please check whether the backend server is running."
        )

    except requests.exceptions.RequestException as error:

        st.error(f"API request failed: {error}")

    except (KeyError, ValueError) as error:

        st.error(f"Invalid API response: {error}")

# --------------------------------
# 7. Footer
# --------------------------------

st.divider()

st.caption(
    "FraudGuard AI | Machine Learning Project"
)

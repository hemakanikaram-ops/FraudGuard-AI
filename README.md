
## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/hemakanikaram-ops/FraudGuard-AI.git
cd FraudGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the dataset

Download the Credit Card Fraud Detection dataset from Kaggle and place `creditcard.csv` inside the `data/` folder.

### 6. Start FastAPI backend

```bash
uvicorn backend.main:app --reload
```

Open API documentation:

http://127.0.0.1:8000/docs

### 7. Start Streamlit dashboard

Open another terminal:

```bash
streamlit run dashboard/app.py
```

### 8. View prediction history

```bash
streamlit run dashboard/history.py
```

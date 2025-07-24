import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# --------- Generate Synthetic Telco-like Data ---------
@st.cache_data
def generate_telco_data(n_samples=3000):
    # Numeric features
    X_num, y = make_classification(n_samples=n_samples, n_features=5,
                                    n_informative=3, n_redundant=1,
                                    weights=[0.73, 0.27], random_state=42)
    
    df = pd.DataFrame(X_num, columns=[
        'MonthlyCharges', 'Tenure', 'TotalCharges', 'NumServices', 'SupportTickets'
    ])
    
    # Add categorical features
    df['gender'] = np.random.choice(['Male', 'Female'], size=n_samples)
    df['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples)
    df['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples)
    df['PaymentMethod'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples)
    df['Churn'] = y
    
    return df

# Load data
data = generate_telco_data()

st.title("📈 Telco Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to churn using logistic regression on synthetic, realistic telco data.")

# Show data
if st.checkbox("🔍 Show Sample Data"):
    st.dataframe(data.head())

# Preprocess
df = data.copy()
df['Churn'] = df['Churn'].astype(int)

y = df['Churn']
X = df.drop('Churn', axis=1)

# Label encode binary categorical features
binary_cols = ['gender']
le_dict = {}
for col in binary_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# One-hot encode multi-class categoricals
multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
X = pd.get_dummies(X, columns=multi_cols, drop_first=True)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
st.subheader("📊 Model Evaluation")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

st.write(f"**Accuracy:** {acc:.4f} | **Precision:** {prec:.4f} | **Recall:** {rec:.4f}")
st.json(report)

# ----------- User Prediction Interface --------------
st.subheader("🧑‍💼 Try it Yourself - Predict Customer Churn")

user_input = {}

user_input['MonthlyCharges'] = st.slider("Monthly Charges", float(data['MonthlyCharges'].min()), float(data['MonthlyCharges'].max()), float(data['MonthlyCharges'].mean()))
user_input['Tenure'] = st.slider("Tenure (months)", float(data['Tenure'].min()), float(data['Tenure'].max()), float(data['Tenure'].mean()))
user_input['TotalCharges'] = st.slider("Total Charges", float(data['TotalCharges'].min()), float(data['TotalCharges'].max()), float(data['TotalCharges'].mean()))
user_input['NumServices'] = st.slider("Number of Services Used", float(data['NumServices'].min()), float(data['NumServices'].max()), float(data['NumServices'].mean()))
user_input['SupportTickets'] = st.slider("Customer Support Tickets", float(data['SupportTickets'].min()), float(data['SupportTickets'].max()), float(data['SupportTickets'].mean()))

user_input['gender'] = st.selectbox("Gender", ['Male', 'Female'])
user_input['InternetService'] = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
user_input['Contract'] = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
user_input['PaymentMethod'] = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

# Encode and prepare input
user_df = pd.DataFrame([user_input])
user_df['gender'] = le_dict['gender'].transform(user_df['gender'])

user_df = pd.get_dummies(user_df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)
for col in X.columns:
    if col not in user_df.columns:
        user_df[col] = 0  # Add missing columns

user_df = user_df[X.columns]  # Arrange columns
scaled_input = scaler.transform(user_df)

if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)[0]
    st.markdown("### 🔍 Prediction:")
    if prediction == 1:
        st.error("⚠️ High Risk of Churn Detected!")
    else:
        st.success("✅ Customer Likely to Stay")

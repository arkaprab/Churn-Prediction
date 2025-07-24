import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)
n_samples = 800
data = pd.DataFrame({
    'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
    'Tenure': np.random.randint(1, 73, n_samples),
    'NumServicesUsed': np.random.randint(1, 7, n_samples),
    'SupportTickets': np.random.randint(0, 6, n_samples),
    'ContractType': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.2, 0.2]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'None'], n_samples, p=[0.4, 0.5, 0.1]),
    'PaymentMethod': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], n_samples)
})

data['TotalCharges'] = np.round(data['MonthlyCharges'] * data['Tenure'], 2)

data['Churn'] = np.where(
    (data['MonthlyCharges'] > 80) & (data['SupportTickets'] >= 3), 1,
    np.where((data['Tenure'] < 12) & (data['NumServicesUsed'] <= 2), 1, 0)
)

X = data.drop('Churn', axis=1)
y = data['Churn']

X_encoded = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Prediction App")
st.markdown("This app predicts whether a customer is likely to churn based on realistic service details.")

st.markdown("### 📋 Enter Customer Details")

monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 70.0)
tenure = st.slider("Tenure (months)", 1, 72, 24)
total_charges = round(monthly_charges * tenure, 2)
st.markdown(f"**Total Charges (auto-calculated)**: ${total_charges}")

num_services = st.slider("Number of Services Used", 1, 6, 3)
support_tickets = st.slider("Number of Support Tickets", 0, 5, 1)

contract_type = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'None'])
payment_method = st.selectbox("Payment Method", ['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'])

input_dict = {
    'MonthlyCharges': monthly_charges,
    'Tenure': tenure,
    'NumServicesUsed': num_services,
    'SupportTickets': support_tickets,
    'ContractType': contract_type,
    'InternetService': internet_service,
    'PaymentMethod': payment_method,
    'TotalCharges': total_charges
}

user_input = pd.DataFrame([input_dict])
user_input_encoded = pd.get_dummies(user_input)
user_input_encoded = user_input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
user_input_scaled = scaler.transform(user_input_encoded)

if st.button("🔍 Predict Churn"):
    prediction = model.predict(user_input_scaled)[0]
    if prediction == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

accuracy = accuracy_score(y_test, model.predict(X_test))
st.markdown(f"📊 **Model Accuracy**: `{accuracy * 100:.2f}%`")

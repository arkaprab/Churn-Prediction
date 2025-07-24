import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
np.random.seed(42)
n_samples = 500
data = pd.DataFrame({
    'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
    'Tenure': np.random.randint(1, 72, n_samples),
    'TotalCharges': lambda df: np.round(df['MonthlyCharges'] * df['Tenure'], 2),
    'NumServicesUsed': np.random.randint(1, 7, n_samples),
    'SupportTickets': np.random.randint(0, 6, n_samples),
})
data['TotalCharges'] = data['MonthlyCharges'] * data['Tenure']
data['Churn'] = np.where(
    (data['MonthlyCharges'] > 80) & (data['SupportTickets'] > 2), 1,
    np.where((data['Tenure'] < 12) & (data['NumServicesUsed'] < 3), 1, 0)
)

# Features and labels
X = data.drop('Churn', axis=1)
y = data['Churn']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("🧑\u200d💼 Try it Yourself - Predict Customer Churn")

st.markdown("### Enter Customer Details")

monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 50.0)
tenure = st.slider("Tenure (months)", 1, 72, 24)
total_charges = round(monthly_charges * tenure, 2)
st.write(f"Automatically calculated Total Charges: **${total_charges}**")

num_services = st.slider("Number of Services Used", 1, 6, 3)
support_tickets = st.slider("Customer Support Tickets", 0, 5, 1)

# Prepare input
user_input = pd.DataFrame([[monthly_charges, tenure, total_charges, num_services, support_tickets]],
                          columns=X.columns)
user_input_scaled = scaler.transform(user_input)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(user_input_scaled)[0]
    if prediction == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

# Accuracy display
acc = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: **{acc*100:.2f}%**")

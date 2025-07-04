import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges column
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()

# Drop ID column
data = data.drop('customerID', axis=1)

# Map target variable
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
y = data['Churn']
X = data.drop('Churn', axis=1)

# Identify binary categorical columns
binary_cols = []
for col in X.columns:
    if X[col].nunique() == 2 and X[col].dtype == 'object':
        binary_cols.append(col)

# Label encode binary columns
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Identify multiclass categorical columns
multiclass_cols = []
for col in X.columns:
    if X[col].dtype == 'object':
        multiclass_cols.append(col)

# One-hot encode multiclass categorical columns
X = pd.get_dummies(X, columns=multiclass_cols, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

# Evaluation metrics
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)

print()
print("Test Accuracy:", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:", round(rec, 4))
print()
print("Classification Report:\n")
print(classification_report(y_test, pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.yticks([0, 1], ['No Churn', 'Churn'])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        color = 'white' if value > cm.max() / 2 else 'black'
        plt.text(j, i, str(value), ha='center', va='center', color=color)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

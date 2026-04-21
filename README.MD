# 📘 Scikit-learn Learning Journey

## 🚀 Day 1: ML Basics & First Model

### 🧠 What I Learned

* Machine Learning basics (how models learn from data)
* Difference between **features (X)** and **labels (y)**
* Importance of **train-test split**
* Built my first model using **Logistic Regression**
* Evaluated model using **accuracy score**

---

### 💻 Code Example

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample Data
X = [[2], [4], [6], [8]]
y = [0, 0, 1, 1]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
```

---

## 🚀 Day 2: Data Preprocessing

### 🧠 What I Learned

* Importance of **clean and prepared data**
* **Feature scaling** using StandardScaler
* **Encoding categorical data** using LabelEncoder
* Handling **missing values** using SimpleImputer
* Real-world ML pipeline understanding

---

### 💻 Code Examples

#### 🔹 Scaling

```python
from sklearn.preprocessing import StandardScaler

X = [[20, 20000], [30, 40000], [40, 60000]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

---

#### 🔹 Encoding

```python
from sklearn.preprocessing import LabelEncoder

gender = ["Male", "Female", "Male"]

encoder = LabelEncoder()
encoded = encoder.fit_transform(gender)

print(encoded)
```

---

#### 🔹 Missing Values

```python
from sklearn.impute import SimpleImputer
import numpy as np

X = [[1], [2], [np.nan], [4]]

imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)

print(X_filled)
```

---

## 🎯 Key Takeaways

* ML models depend heavily on **data quality**
* Proper preprocessing improves model performance
* Understanding the ML pipeline is essential before moving to advanced topics

---

## 📌 Next Plan

* Dive deeper into **Logistic Regression**
* Understand **decision boundaries**
* Explore more ML models

---

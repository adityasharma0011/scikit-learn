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

------------------------------------------------------------------------------------------------------------------------------------



## 🚀 Day 3: Logistic Regression & Error Analysis

### 🧠 What I Learned

* How a classification model (Logistic Regression) makes predictions
* Concept of **decision boundary** (how model separates classes)
* Importance of checking **model errors**, not just accuracy
* Understanding **Confusion Matrix** to analyze prediction results
* Learned how **outliers and inconsistent data** affect model performance

---

### 💻 Code Example

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Sample Data
X = [[2], [4], [6], [8], [10]]
y = [0, 0, 1, 1, 1]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

### 🔍 Confusion Matrix Explained

```text
[[TN  FP]
 [FN  TP]]
```

* **TN (True Negative):** Correctly predicted class 0
* **FP (False Positive):** Incorrectly predicted class 1
* **FN (False Negative):** Incorrectly predicted class 0
* **TP (True Positive):** Correctly predicted class 1

👉 Diagonal values = correct predictions
👉 Off-diagonal values = errors

---

### ⚠️ Key Insight

* Models do not always predict perfectly
* Errors can occur due to:

  * insufficient data
  * inconsistent patterns
  * outliers

---

### 📌 Experiment Done

Tested model behavior with unusual data:

```python
X = [[1], [2], [3], [100]]
y = [0, 0, 1, 1]
```

👉 Observed how **outliers affect predictions**

---

### 🎯 Key Takeaways

* Understanding **why a model makes mistakes** is more important than just accuracy
* Confusion Matrix helps in identifying **type of errors**
* Data quality directly impacts model performance

---

### 📌 Next Plan

* Learn **Decision Tree model**
* Compare it with Logistic Regression
* Understand different model behaviors



## 🚀 Day 4: Decision Tree & Model Comparison

### 🧠 What I Learned

* How **Decision Tree** works using rule-based splitting (if-else logic)
* Difference between **Decision Tree** and **Logistic Regression**
* Concept of **overfitting** in Decision Trees
* How different models behave on:

  * clean data
  * inconsistent data
  * outliers
* Importance of **model comparison and behavior analysis**

---

## 🌳 Decision Tree Intuition

Decision Tree creates rules like:

```text
if feature < threshold:
    class = 0
else:
    class = 1
```

👉 Unlike Logistic Regression (which creates a smooth boundary), Decision Tree creates **hard splits**

---

## 💻 Code Example

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X = [[2],[4],[6],[8],[10]]
y = [0,0,1,1,1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Prediction:", y_pred)
print("Actual:", y_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## 🔍 Experiments Performed

### 🔹 1. Logistic vs Decision Tree

* Compared both models on same dataset
* Observed:

  * Logistic → smooth prediction
  * Tree → exact rule-based prediction

👉 Insight:
**Decision Tree handles sharp boundaries better**

---

### 🔹 2. Inconsistent Data Test

```python
X = [[2],[4],[6],[8]]
y = [0,1,1,1]
```

👉 Observation:

* Different models produced different errors
* Logistic → False Positive
* Tree → False Negative

👉 Insight:
**Models interpret inconsistent data differently**

---

### 🔹 3. Outlier Experiment

```python
X = [[1],[2],[3],[100],[200]]
y = [0,0,1,1,1]
```

👉 Observation:

* Logistic → boundary shift
* Tree → rule-based split

👉 Insight:
**Decision Tree is less sensitive to outliers compared to Logistic Regression**

---

### 🔹 4. Overfitting Case

```text
Prediction: [0 0]
Actual:     [0 1]

Confusion Matrix:
[[1 0]
 [1 0]]
```

👉 Observation:

* Model predicted only class 0
* Failed to detect class 1

👉 Insight:
**Decision Tree overfits and fails to generalize**

---

### 🔹 5. Depth Control

```python
DecisionTreeClassifier(max_depth=1)
```

👉 Observation:

* Lower depth → simple model
* Higher depth → complex model (overfitting risk)

👉 Insight:
**Controlling depth helps prevent overfitting**

---

## ⚠️ Key Learnings

* Different models = different decision strategies
* Data quality directly affects model performance
* Overfitting is a major issue in Decision Trees
* Confusion Matrix helps identify type of errors
* Testing with different datasets is crucial

---

## 🎯 Key Takeaways

* Decision Tree = rule-based model
* Logistic Regression = smooth boundary model
* No model is always correct → depends on data
* Understanding **why a model fails** is more important than accuracy

---

## 📌 Next Plan

* Start mini project using learned concepts
* Explore real datasets
* Move towards advanced topics (RAG, embeddings)

---



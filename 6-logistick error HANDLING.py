# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# X = [[2], [4], [6], [8], [10]]
# y = [0, 0, 1, 1, 1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# print("Prediction:", model.predict(X_test))
# print("Actual:", y_test)

# from sklearn.metrics import confusion_matrix

# y_pred = model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))






# # simple logistic regression example with error checking

# #  experiment ----2-------- changing data to check for errors in logistic regression model

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# X = [[1], [2], [3], [4], [5], [6]]
# y = [0, 0, 0, 1, 1, 1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# print("Prediction:", model.predict(X_test))
# print("Actual:", y_test)

# from sklearn.metrics import confusion_matrix

# y_pred = model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))





# # EXPERIMENT --3--------------TAKINNG CONFUSION DATA TO CHECK FOR ERRORS IN LOGISTIC REGRESSION MODEL


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# change data to check for errors in logistic regression model
X = [[1],[3], [4], [5], [6]]
y = [0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Prediction:", model.predict(X_test))
print("Actual:", y_test)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))





# # EXPERIMENT--4---------OUTLIER TEST-----------------------------------------

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# # change data to check for errors in logistic regression model
# X = [[1],[2],[3], [100],[200]]
# y = [0 , 1 , 0 , 1,1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# print("Prediction:", model.predict(X_test))
# print("Actual:", y_test)

# from sklearn.metrics import confusion_matrix

# y_pred = model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
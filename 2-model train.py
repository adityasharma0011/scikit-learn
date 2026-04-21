from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[2], [4], [6], [8],[10],[12]]   # input
y = [0, 0, 1, 1,1,1]          # output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

model = LogisticRegression()

model.fit(X_train, y_train)   # training

prediction = model.predict(X_test)

print(prediction)


print(accuracy_score(y_test, prediction))
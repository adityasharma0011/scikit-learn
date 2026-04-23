# #: Decision Tree (Easy + Powerful)
# 🧠 Decision Tree kya hota hai?

# # 👉 Ye model asks questions  (if-else)    
# [if marks < 5:
#     Fail
# else:
#   Pass   ]


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X = [[2], [4], [6], [8], [10]]
y = [0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Prediction:", model.predict(X_test))
print("Actual:", y_test)



from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))




# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# plot_tree(model, filled=True)
# plt.show()



# confusing data to check for errors in decision tree model

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X = [[2], [4], [6], [8]]
y = [0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Prediction:", model.predict(X_test))
print("Actual:", y_test)



from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))







#  overfitting checking in decision tree



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X = [[1],[2],[3],[4],[5],[6]]
y = [0,1,0,1,0,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Prediction:", model.predict(X_test))
print("Actual:", y_test)



from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

from sklearn.impute import SimpleImputer
import numpy as np

X = [[1], [2], [np.nan], [4]]

imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)

print(X_filled)
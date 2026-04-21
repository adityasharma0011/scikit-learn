from sklearn.preprocessing import LabelEncoder

gender = ["Male", "Female", "Male"]

encoder = LabelEncoder()
encoded = encoder.fit_transform(gender)

print(encoded)
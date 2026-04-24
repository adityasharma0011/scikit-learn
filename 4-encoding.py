from sklearn.preprocessing import LabelEncoder
import pandas as pd

# gender = ["Male", "Female", "Male"]
# # encoder = LabelEncoder()
# # encoded = encoder.fit_transform(gender)
# # print(encoded)



df = pd.read_csv("sample_data.csv")
df_label = df.copy()

le = LabelEncoder()
df_label['Gender_Encoded'] = le.fit_transform(df_label['Gender'])
df_label['Passed_Encoded'] = le.fit_transform(df_label['Passed'])

print('\nlabel encoded data')
print(df_label[['Name','Gender', 'Gender_Encoded','Passed','Passed_Encoded' ]])

#  above this was all about label encoding, now we will see one hot encoding




df_encoded = pd.get_dummies(df, columns=['City'])
print('\none-hot encoded data(City)')
print(df_encoded)

# this is converting the 'City' column into bollean ans task  is to convert into  binary columns




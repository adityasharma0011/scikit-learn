import pandas as pd
 

data = { 
  'name': ['pavan', 'aman', ' adi', ] ,
  'age': [25, None, 30],
  'salary': [50000, 60000, None]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)


# find missing values

print(df.isnull().sum())

#  drop rows with missing values

df_drop = df.dropna()
print(df_drop)


# fill missing values with mean-average value

df.fillna({'age': df['age'].mean()}, inplace=True)
df.fillna({'salary': df['salary'].mean()}, inplace=True)
print(df)


# important things---------
# 1-- never drop data blindly, first try to fill data with mean,meaadian
# 2-- if working with numeric data always fill with mean or median, if working with categorical data fill with mode
# 3-- check how much % data is missing before cleaning data.


print(df.isnull().mean()*100)
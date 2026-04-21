from sklearn.preprocessing import StandardScaler

X = [[20, 20000], [30, 40000], [40, 60000]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
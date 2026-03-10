from sklearn.preprocessing import StandardScaler
def scale_data(x):
    scaler = StandardScaler()
    x_scaler = scaler.fit_transform(x)
    return x_scaler,scaler
from data.load_data import load_faces
from preprocessing.preprocess import scale_data
from models.pca_model import apply_pca
from models.train_model import train_svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config
from visualization.visualize import show_simple_faces, show_mean_variance, plot_variance, show_eigenfaces
import numpy as np
x , y = load_faces()
show_simple_faces(x, y)
x_scaler ,scaler = scale_data(x)

mean_face = scaler.mean_

show_mean_variance(mean_face)
x_pca ,pca_full = apply_pca(x_scaler,config.N_COMPONENTS)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100

plot_variance(cumulative_variance)
show_eigenfaces(pca_full)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.25, random_state=42, stratify=y)
model = train_svm(x_train,y_train,config.SVM_C,config.SVM_GAMMA,config.SVM_KERNEL)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test,y_pred)
print("Accuracy:",acc * 100)



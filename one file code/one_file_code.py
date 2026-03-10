from cProfile import label

from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = fetch_olivetti_faces()
x = data.data
y = data.target

print("total images:", x.shape[0])
print("pixels per img:", x.shape[1])
print("number of people:", len(np.unique(y)))
print('----------------------------------------------------\n')
fig, axes = plt.subplots(3, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x[i * 10].reshape(64, 64), cmap='gray')
    ax.set_title(f"person {y[i * 10]}")
    ax.axis('off')
plt.suptitle("Sample Faces", fontsize=16)
plt.tight_layout()
plt.show()
print('----------------------------------------------------\n')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

mean_shape = scaler.mean_.reshape(64, 64)
plt.imshow(mean_shape, cmap='gray')
plt.title('The average face is mean face')
plt.axis('off')
plt.show()
print('-----------------------------------------------')

from sklearn.decomposition import PCA

pca_full = PCA()
pca_full.fit(x_scaler)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100

print('----------------------------------------------------------------------')
plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance, linewidth=2)
plt.axhline(y=95, linestyle='--', color='r', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.title('How Many Components Do We Need?')
plt.legend()
plt.grid(True)
plt.show()
n_95 = np.argmax(cumulative_variance >= 95) + 1
print(f'component needed for 95% variance : {n_95}')
# print(cumulative_variance)
n_components = 100
pca = PCA(n_components)
x_pca = pca.fit_transform(x_scaler)
print(x_pca)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    eigenface = pca.components_[i].reshape(64, 64)
    variance = pca.explained_variance_ratio_[i] * 100
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f'PC{i + 1}\n({variance:.1f}%var)')
    ax.axis('off')
plt.suptitle("Top 10 eigenfaces ", fontsize=16)
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.25, random_state=42, stratify=y)
'''
stratify=y  → VERY IMPORTANT!
              Ensures each person has EQUAL representation
              in both train and test sets.

              Without stratify:
                  Training might get all 10 photos of Person 5
                  Testing gets 0 photos of Person 5
                  Model never saw Person 5 → can't recognize them!

              With stratify:
                  Each person gets ~7-8 in training, ~2-3 in testing
                  Fair for everyone!
'''
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy : {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))
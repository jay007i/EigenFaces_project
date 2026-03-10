# 🧠 Face Recognition using PCA (Eigenfaces)

## 📌 Project Overview
Built a face recognition system using Principal Component Analysis (PCA) 
that reduces 4096-dimensional face data to just 100 dimensions and achieves 
96%+ accuracy using SVM classifier.

## 🔍 What This Project Does
- Loads 400 face images (40 people × 10 images each)
- Applies PCA to extract "Eigenfaces" (principal components)
- Reduces dimensions from 4096 → 100 (40x compression!)
- Trains an SVM classifier for face recognition
- Achieves 96% accuracy on unseen faces
- Visualizes eigenfaces, reconstructions, and 2D projections

## 🛠️ Tech Stack
- Python
- NumPy
- Scikit-learn (PCA, SVM, StandardScaler)
- Matplotlib

## 📊 Key Results
| Metric | Without PCA | With PCA |
|--------|-------------|----------|
| Features | 4096 | 100 |
| Accuracy | 95.0% | 96.0% |
| Training Time | 0.450s | 0.035s |

## 📸 Visualizations

### Eigenfaces
![Eigenfaces](images/eigenfaces.png)

### Face Reconstruction with Increasing Components
![Reconstruction](images/reconstruction.png)

### 2D PCA Projection
![2D Plot](images/2d_projection.png)

## 🧮 How PCA Works (Brief)
1.

import matplotlib.pyplot as plt
from preprocessing.preprocess import scale_data
def show_simple_faces(x,y):
    fig, axes = plt.subplots(3, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i * 10].reshape(64, 64), cmap='gray')
        ax.set_title(f"person {y[i * 10]}")
        ax.axis('off')
    plt.suptitle("Sample Faces", fontsize=16)
    plt.tight_layout()
    plt.show()
def show_mean_variance(scaler):
    plt.imshow(scaler.reshape(64, 64),cmap='gray')
    plt.title("Mean Face")
    plt.axis('off')
    plt.show()


def plot_variance(cumulative_variance):
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_variance, linewidth=2)
    plt.axhline(y=95, linestyle='--', color='r', label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained (%)')
    plt.title('How Many Components Do We Need?')
    plt.legend()
    plt.grid(True)
    plt.show()

def show_eigenfaces(pca):

    fig, axes = plt.subplots(2,5, figsize=(15,6))

    for i, ax in enumerate(axes.flat):

        eigenface = pca.components_[i].reshape(64,64)

        variance = pca.explained_variance_ratio_[i] * 100

        ax.imshow(eigenface, cmap='gray')
        ax.set_title(f'PC{i+1}\n({variance:.1f}%)')
        ax.axis('off')

    plt.suptitle("Top 10 Eigenfaces")
    plt.tight_layout()
    plt.show()
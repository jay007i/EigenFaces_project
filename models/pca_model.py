from sklearn.decomposition import PCA


def apply_pca(x,n_components):
    pca_full = PCA(n_components=n_components)
    x_pca = pca_full.fit_transform(x)

    return x_pca,pca_full





'''
from sklearn.decomposition import PCA
pca_full = PCA()
pca_full.fit(x_scaler)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100

'''
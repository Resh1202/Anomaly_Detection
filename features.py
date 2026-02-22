from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

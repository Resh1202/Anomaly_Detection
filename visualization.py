import matplotlib.pyplot as plt

def plot_score_distribution(scores):
    plt.figure()
    plt.hist(scores, bins=50)
    plt.title("Anomaly Score Distribution")
    return plt

def plot_pca_projection(X_pca, labels):
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("2D PCA Projection")
    return plt

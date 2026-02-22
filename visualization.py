import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca(data, labels):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:,0], pca_data[:,1], c=labels, cmap='coolwarm')
    ax.set_title("PCA Anomaly Visualization")
    st.pyplot(fig)

def plot_scores(scores):
    st.subheader("Anomaly Scores")
    st.line_chart(scores)
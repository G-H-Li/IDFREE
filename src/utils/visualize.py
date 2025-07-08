import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca_decomposition(embeddings, n_dims=64):
    # Apply PCA
    pca = PCA(n_components=n_dims, random_state=999)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def visualize_modal_embeddings(id_emb, v_emb, t_emb, title_name, modals=None):
    if modals is None:
        modals = ['Id', 'Video', 'Text']
    num = id_emb.shape[0]
    dim = id_emb.shape[1]
    v_de_emb = pca_decomposition(v_emb, dim)
    t_de_emb = pca_decomposition(t_emb, dim)
    embeddings = np.concatenate([id_emb, v_de_emb, t_de_emb], axis=0)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=999, perplexity=50, learning_rate=100)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot the results
    plt.figure(figsize=(10, 10))
    colors = ['red', 'green', 'blue']
    for i in range(0, num * 3, num):
        k = i // num
        plt.scatter(reduced_embeddings[i: i+num, 0], reduced_embeddings[i: i+num, 1], c=colors[k], label=modals[k])
    plt.legend()
    plt.title(title_name)
    plt.show()
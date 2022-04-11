import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def plot_3d_PCA(X, y, method, prop):
    pca = PCA(n_components=3)
    new_com = pca.fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(new_com[y == 0, 0], new_com[y == 0, 1], new_com[y == 0, 2], color='b', alpha=0.2)
    ax.scatter(new_com[y == 1, 0], new_com[y == 1, 1], new_com[y == 1, 2], color='r', alpha=1)
    path = f'reports/figures/pca/pca_3d_{method}_prop_{prop}.png'
    plt.savefig(path)
    print(f"Saved figure {path}")
    
    
def plot_2d_PCA(X, y, method, prop):
    pca = PCA(n_components=2)
    new_com = pca.fit_transform(X)
    fig = plt.figure()
    plt.scatter(new_com[y == 0, 0], new_com[y == 0, 1], color='b', alpha=0.2)
    plt.scatter(new_com[y == 1, 0], new_com[y == 1, 1], color='r', alpha=1)
    path = f'reports/figures/pca/pca_2d_{method}_prop_{prop}.png'
    plt.savefig(path)
    print(f"Saved figure {path}")
    
def plot_corr(X, y, method, prop):
    
    df = pd.concat([X, y], axis=1)
    print(df)
    corr = df.corr(method='kendall')
    sc = 1
    plt.figure(figsize=(sc*10, sc*10))
    plt.imshow(corr, origin='lower')#, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.xticks(np.arange(len(df.columns)), df.columns, fontsize=9*sc, rotation=90)
    plt.yticks(np.arange(len(df.columns)), df.columns, fontsize=9*sc)
    plt.colorbar()
    path = f'reports/figures/corr/corr_{method}_prop_{prop}.png'
    plt.savefig(path)
    print(f"Saved figure {path}")
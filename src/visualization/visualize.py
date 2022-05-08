import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
import pandas as pd
import numpy as np
import scikitplot as skplt

def plot_3d_PCA(X, y, name):
    pca = PCA(n_components=3)
    new_com = pca.fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(new_com[y == 0, 0], new_com[y == 0, 1], new_com[y == 0, 2], color='b', alpha=0.2)
    ax.scatter(new_com[y == 1, 0], new_com[y == 1, 1], new_com[y == 1, 2], color='r', alpha=1)
    path = f'reports/figures/{name}/pca_3d.png'
    plt.savefig(path)
    #print(f"Saved figure {path}")
    
    
def plot_2d_PCA(X, y, name):
    pca = PCA(n_components=2)
    new_com = pca.fit_transform(X)
    fig = plt.figure()
    plt.scatter(new_com[y == 0, 0], new_com[y == 0, 1], color='b', alpha=0.2)
    plt.scatter(new_com[y == 1, 0], new_com[y == 1, 1], color='r', alpha=1)
    path = f'reports/figures/{name}/pca_2d.png'
    plt.savefig(path)
    #print(f"Saved figure {path}")
    
def plot_corr(X, y, name):
    
    df = pd.concat([X, y], axis=1)
    corr = df.corr(method='kendall')
    sc = 1
    plt.figure(figsize=(sc*10, sc*10))
    plt.imshow(corr, origin='lower')#, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.xticks(np.arange(len(df.columns)), df.columns, fontsize=9*sc, rotation=90)
    plt.yticks(np.arange(len(df.columns)), df.columns, fontsize=9*sc)
    plt.colorbar()
    path = f'reports/figures/{name}/corr.png'
    plt.savefig(path)
    #print(f"Saved figure {path}")
    
def plot_roc(probs, y, name):
    
    skplt.metrics.plot_roc(np.array(y), probs)
    
    path = f'reports/figures/{name}/roc.png'
    plt.savefig(path)
    
def plot_conf_matr(preds, y, name):
    
    skplt.metrics.plot_confusion_matrix(y, preds)
    
    path = f'reports/figures/{name}/conf_matr.png'
    plt.savefig(path)
    
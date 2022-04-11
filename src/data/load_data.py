import pandas as pd
from sklearn.model_selection import train_test_split
from src.features.make_feats import make_feats

def load_data(datapath: str, descpath: str, test_size: float, namevars: bool, make_feats_method: str, propFeats: bool):
    df = pd.read_csv(datapath)
    y = df.Y
    X = df.drop(columns=['Y'])
    X = make_feats(X, make_feats_method, propFeats)
    if namevars and make_feats_method == 'old':
        desc = pd.read_csv(descpath, sep=';')
        X[desc['OPIS'].values] = X
        X = X.drop(columns=desc['NAZWA'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0, stratify=y) 
    return X_train, y_train, X_test, y_test

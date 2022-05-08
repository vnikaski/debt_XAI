import pandas as pd
from sklearn.model_selection import train_test_split
from src.features.make_feats import make_feats

def load_data(datapath: str, descpath: str, test_size: float, namevars: bool, make_feats_method: str, propFeats: bool):
    col = ['Y', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X78', 'X199', 'X268', 'X269', 'X270']
    col = ['Y', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X44', 'X46', 'X78', 'X79', 'X81', 'X131', 'X190', 'X194', 'X195', 'X199', 'X210', 'X268', 'X269', 'X270']
    df = pd.read_csv(datapath)#[col]
    y = df.Y
    X = df.drop(columns=['Y'])
    X = make_feats(X, make_feats_method, propFeats)
    if namevars and make_feats_method == 'old':
        desc = pd.read_csv(descpath, sep=';')
        X[desc['OPIS'].values] = X
        X = X.drop(columns=desc['NAZWA'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0, stratify=y) 
    return X_train, y_train, X_test, y_test

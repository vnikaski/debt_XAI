import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessors.new_feats import new_feats


def load_data(datapath: str, descpath: str, test_size: float, namevars=True):
    df = pd.read_csv(datapath)
    df = new_feats(df)
    if not namevars:
        desc = pd.read_csv(descpath, sep=';')
        df[desc['OPIS'].values] = df
        df = df.drop(columns=desc['NAZWA'].values)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0], test_size=test_size, random_state=0, shuffle=True, stratify=df[df.columns[0]])
    return X_train, y_train, X_test, y_test

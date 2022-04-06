import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessors.new_feats import new_feats


def load_data(datapath: str, descpath: str, test_size: float, namevars=True):
    df = pd.read_csv(datapath)
    df, y = new_feats(df)
    # Dla nowych krótszych nazw to chyba będzie niepotrzebne
    '''if not namevars:
        desc = pd.read_csv(descpath, sep=';')
        df[desc['OPIS'].values] = df
        df = df.drop(columns=desc['NAZWA'].values)'''
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, shuffle=True, stratify=y) # random_state=0, 
    return X_train, y_train, X_test, y_test

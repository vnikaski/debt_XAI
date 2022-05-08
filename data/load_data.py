import pandas as pd
from sklearn.model_selection import train_test_split
import warnings


def load_data(datapath: str, descpath: str, test_size: float, val_size: float, namevars=True):
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    df = pd.read_csv(datapath)
    if namevars:
        desc = pd.read_csv(descpath, sep=';')
        df[desc['OPIS'].values] = df
        df = df.drop(columns=desc['NAZWA'].values)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0], test_size=(test_size+val_size), random_state=0, shuffle=True, stratify=df[df.columns[0]])
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size/(test_size+val_size), random_state=0, shuffle=True)
    return X_train, y_train, X_test, y_test, X_val, y_val

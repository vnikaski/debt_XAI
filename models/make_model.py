from typing import List
from sklearn.ensemble import RandomForestClassifier

def make_model(model_type: str) -> List[None]:
    
    return [RandomForestClassifier(n_jobs=10)]

from typing import List
from imblearn.over_sampling import SMOTE, RandomOverSampler

def make_preprocessor(processes: List[str]) -> List[tuple]:
    
    return [('smote', SMOTE())]

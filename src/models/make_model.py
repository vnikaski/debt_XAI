from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def make_model(model_type: str, learning_rate: float, n_estimatiors: int, max_depth: int, class_weight: str) -> Any:
    if model_type is None:
        raise ValueError("Model not compatible")
        
    if 'lr' in model_type:
        return LogisticRegression(random_state=0,
                                  class_weight=class_weight, max_iter=800)

    if 'rf' in model_type:
        return RandomForestClassifier(n_estimators=n_estimatiors,
                                      max_depth=max_depth,
                                      random_state=0,
                                      class_weight=class_weight)

    if 'gb' in model_type:
        return GradientBoostingClassifier(learning_rate=learning_rate,
                                          n_estimators=n_estimatiors,
                                          max_depth=max_depth,
                                          random_state=0)

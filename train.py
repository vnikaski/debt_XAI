import argparse
import numpy as np

#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline # tu można dorzucić resampling a w sklearn nie
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from data.load_data import load_data
from models.make_model import make_model
from preprocessors.make_preprocessor import make_preprocessor

parser = argparse.ArgumentParser()

parser.add_argument('--prep', choices=['ROS', 'SMOTE', 'RUS', 'weighted', 'normalize', 'pca', 'svd'], nargs="+")
parser.add_argument('classifier', choices=['lr', 'rf', 'gb'])
parser.add_argument('--dataPath', type=str, default='temat_3_dane.csv')
parser.add_argument('--descPath', type=str, default='temat_3_opis_zmiennych.csv')
parser.add_argument('--savePath', type=str, default=None)
parser.add_argument('--leaveCodeNames', action='store_false')
parser.add_argument('--testSize', type=float, default=0.1)

args = parser.parse_args()

pipe_steps = make_preprocessor(args.prep)
model = make_model(args.classifier)[0]

pipe_steps.append(('model', model))
pipe_cv =  Pipeline(pipe_steps)

X, y, X_test, y_test = load_data(args.dataPath, args.descPath, args.testSize, args.leaveCodeNames)

scores_cv = cross_val_score(pipe_cv, X, y, cv=5, scoring='f1')

print(f"CV f1 score {np.mean(scores_cv)}")

pipe_test = model # Pipeline(pipe_steps)
pipe_test.fit(X, y)
preds = pipe_test.predict(X_test)

scores_test = f1_score(y_test, preds)
print(f"TEST f1 score {scores_test}")

# saving the model
if args.savePath is not None:
    pass

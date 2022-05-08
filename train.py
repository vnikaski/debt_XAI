import argparse
import numpy as np
import os
import joblib

from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from src.visualization.visualize import plot_3d_PCA, plot_2d_PCA, plot_corr, plot_roc
from src.data.load_data import load_data
from src.models.make_model import make_model
from src.preprocessors.make_preprocessor import make_preprocessor
import src.features.make_feats as make_feats
dic_method = make_feats.dic_method
parser = argparse.ArgumentParser()

parser.add_argument('--prep', choices=['ROS', 'SMOTE', 'RUS', 'normalize', 'pca', 'svd'], nargs="+")
parser.add_argument('classifier', choices=['lr', 'rf', 'gb'])
parser.add_argument('--dataPath', type=str, default='data/temat_3_dane.csv')
parser.add_argument('--descPath', type=str, default='data/temat_3_opis_zmiennych.csv')
parser.add_argument('--makeFeatsMethod', type=str, default='lin', choices=list(dic_method.keys()))
parser.add_argument('--savePath', type=str, default=None)
parser.add_argument('--leaveCodeNames', action='store_true')
parser.add_argument('--oldFeats', action='store_true')
parser.add_argument('--propFeats', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--testSize', type=float, default=0.1)
parser.add_argument('--osRate', type=float, default=1) # 1 = no oversampling, 2 = twice as many, etc
parser.add_argument('--usRate', type=float, default=1) # 1 = no undersampling, 2 = half of neg samples, etc
parser.add_argument('--nComponents', type=int, default=50)
parser.add_argument('--nFolds', type=int, default=5)
parser.add_argument('--learningRate', type=float, default=0.001) # good default LR for DL, not sure about less complex models
parser.add_argument('--nEstimators', type=float, default=100)
parser.add_argument('--maxDepth', type=float, default=None)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--classWeight', type=str, choices=['balanced, balanced_subsample'], default=None)

args = parser.parse_args()

name = f'model_{args.classifier}_prep_{args.prep[0]}_feat_{args.makeFeatsMethod}_prop_{args.propFeats}_th_{args.threshold}'
print(name)

os.makedirs(f'reports/figures/{name}', exist_ok=True)

X, y, X_test, y_test = load_data(args.dataPath, args.descPath, args.testSize, args.leaveCodeNames, args.makeFeatsMethod, args.propFeats)     

if args.visualize:
    plot_3d_PCA(X, y, name)
    plot_2d_PCA(X, y, name)
    plot_corr(X, y, name)
    
preprocesses = make_preprocessor(processes=args.prep,
                                 os_rate=args.osRate,
                                 us_rate=args.usRate,
                                 cur_ratio=y.sum()/(len(y)-y.sum()),
                                 n_components=args.nComponents)

model = make_model(model_type=args.classifier,
                   learning_rate=args.learningRate,
                   n_estimatiors=args.nEstimators,
                   max_depth=args.maxDepth,
                   class_weight=args.classWeight)
preprocesses.append(('model', model))

pipe_steps=preprocesses

pipe =  Pipeline(pipe_steps)
scores_cv = cross_val_score(pipe, X, y, cv=args.nFolds, scoring='f1')

print(f"CV f1 score {np.mean(scores_cv)}")

pipe.fit(X, y)
probs = pipe.predict_proba(X_test)
preds = (probs[:, 1] > args.threshold) * 1

if args.visualize:
    plot_roc(probs, y_test, name)
    
scores_test = f1_score(y_test, preds)
print(f"TEST f1 score {scores_test}\n\n")
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
space = 10
print(f"CONFUSION MATRIX\n{tn} {fn}\n {fp} {tp}")

spec = tn / (tn+fp)
sens = tp / (tp + fn)

print("SPEC", spec)
print("SENS", sens)


# saving the model
if args.savePath is not None:
    joblib.dump(pipe, f'{args.savePath}/{name}.joblib')
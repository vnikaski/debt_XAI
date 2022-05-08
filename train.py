import argparse
from imblearn.pipeline import Pipeline

from data.load_data import load_data
from models.make_model import make_model
from preprocessors.make_preprocessor import make_preprocessor

parser = argparse.ArgumentParser()

parser.add_argument('--prep', choices=['ROS', 'SMOTE', 'RUS', 'normalize', 'pca', 'svd'], nargs="+")
parser.add_argument('classifier', choices=['lr', 'rf', 'gb'])
parser.add_argument('--dataPath', type=str, default='temat_3_dane.csv')
parser.add_argument('--descPath', type=str, default='temat_3_opis_zmiennych.csv')
parser.add_argument('--savePath', type=str, default=None)
parser.add_argument('--leaveCodeNames', action='store_false')
parser.add_argument('--testSize', type=float, default=0.1)
parser.add_argument('--osRate', type=float, default=1) # 1 = no oversampling, 2 = twice as many, etc
parser.add_argument('--usRate', type=float, default=1) # 1 = no undersampling, 2 = half of neg samples, etc
parser.add_argument('--nComponents', type=int, default=50)
parser.add_argument('--learningRate', type=float, default=0.001) # good default LR for DL, not sure about less complex models
parser.add_argument('--nEstimators', type=float, default=100)
parser.add_argument('--maxDepth', type=float, default=None)
parser.add_argument('--classWeight', type=str, choices=['balanced, balanced_subsample'], default=None)

args = parser.parse_args()

X, y, X_test, y_test = load_data(args.dataPath, args.descPath, args.testSize, args.leaveCodeNames)

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
pipe = Pipeline(pipe_steps)

pipe.fit(X, y)

# saving the model
if args.savePath is not None:
    pass

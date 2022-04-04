import argparse
from sklearn.pipeline import Pipeline

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

preprocesses = make_preprocessor(args.prep)
model = make_model(args.classifier)

pipe_steps = preprocesses.append(('model', model))
pipe = Pipeline(pipe_steps)

X, y, X_test, y_test = load_data(args.dataPath, args.descPath, args.testSize, args.leaveCodeNames)

pipe.fit(X, y)

# saving the model
if args.savePath is not None:
    pass

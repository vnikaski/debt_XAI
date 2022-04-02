import argparse
from sklearn.pipeline import Pipeline

from data.load_data import load_data
from models.make_model import make_model
from preprocessors.make_preprocessor import make_preprocessor

parser = argparse.ArgumentParser()

parser.add_argument('--prep', choices=['ROS', 'SMOTE', 'RUS', 'weighted', 'normalize', 'pca', 'svd'], nargs="+")
parser.add_argument('classifier', choices=['lr', 'rf', 'gb'])
parser.add_argument('--datapath', type=str, default='temat_3_dane.csv')
parser.add_argument('--descpath', type=str, default='temat_3_opis_zmiennych.csv')
parser.add_argument('--savepath', type=str, default=None)

args = parser.parse_args()

preprocesses = make_preprocessor(args.prep)
model = make_model(args.classifier)

pipe_steps = preprocesses.append(('model', model))

pipe = Pipeline(pipe_steps)

X, y, X_test, y_test = load_data(args.datapath, args.descpath)

pipe.fit(X, y)

# saving the model
if args.savepath is not None:
    pass

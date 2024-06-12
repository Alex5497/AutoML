from pycaret.classification import *
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

experiment = setup(data, target='species')

best_model = compare_models()

final_model = finalize_model(best_model)

save_model(final_model, 'iris_species_classification_model')
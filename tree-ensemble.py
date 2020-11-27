from dataset.adult import Adult
from os.path import exists
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


ds_train = Adult(None, train=True)
ds_test = Adult(None, train=False)

X_train, y_train = ds_train.pandas()
X_test, y_test = ds_test.pandas()

model = CatBoostClassifier(
    random_seed=42,
    custom_loss=['Accuracy'],
    iterations=10000,
    logging_level='Verbose',
)

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]


if not exists('tree.model'):

    model.fit(X_train, 
            y_train, 
            cat_features=categorical_features_indices, 
            eval_set=(X_test, y_test))

    model.save_model('tree.model')
else:
    model.load_model('tree.model')

print('test accuracy:', accuracy_score(model.predict(X_test), y_test))
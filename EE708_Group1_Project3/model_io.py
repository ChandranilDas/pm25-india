"""Numeric HDF5 storage for a scikit-learn random forest (not a Keras model)."""
import json
import h5py
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, _tree

def save_model(model, path):
    with h5py.File(path, 'w') as f:
        f.attrs['format'] = 'sklearn_random_forest_numeric_hdf5_v1'
        f.attrs['sklearn_version'] = sklearn.__version__
        f.attrs['params'] = json.dumps(model.get_params())
        f.attrs['features'] = json.dumps(list(model.feature_names_in_))
        f.attrs['n_features'] = model.n_features_in_
        f.attrs['n_outputs'] = model.n_outputs_
        for i, estimator in enumerate(model.estimators_):
            g = f.create_group(f'trees/{i}')
            state = estimator.tree_.__getstate__()
            g.attrs['max_depth'] = state['max_depth']
            g.attrs['node_count'] = state['node_count']
            g.create_dataset('nodes', data=state['nodes'], compression='gzip')
            g.create_dataset('values', data=state['values'], compression='gzip')

def load_model(path):
    with h5py.File(path, 'r') as f:
        if f.attrs['format'] != 'sklearn_random_forest_numeric_hdf5_v1':
            raise ValueError('Unsupported model format')
        if f.attrs['sklearn_version'] != sklearn.__version__:
            raise RuntimeError('Use scikit-learn==' + f.attrs['sklearn_version'])
        model = RandomForestRegressor(**json.loads(f.attrs['params']))
        model.n_features_in_ = int(f.attrs['n_features'])
        model.n_outputs_ = int(f.attrs['n_outputs'])
        model.feature_names_in_ = np.array(json.loads(f.attrs['features']), dtype=object)
        model.estimators_ = []
        for i in range(model.n_estimators):
            g = f[f'trees/{i}']
            tree = _tree.Tree(model.n_features_in_, np.ones(model.n_outputs_, dtype=np.intp), model.n_outputs_)
            tree.__setstate__({'max_depth':int(g.attrs['max_depth']), 'node_count':int(g.attrs['node_count']), 'nodes':g['nodes'][:], 'values':g['values'][:]})
            estimator = DecisionTreeRegressor()
            estimator.tree_ = tree
            estimator.n_features_in_ = model.n_features_in_
            estimator.n_outputs_ = model.n_outputs_
            model.estimators_.append(estimator)
        return model

import numpy as np
import sklearn.metrics as mt

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num=20)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5, 10, 15, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# scoring metric
scoring = 'neg_median_absolute_error'  # 'mean_squared_error' 'explained_variance' 'r2'

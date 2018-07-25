from eda_support_func import *
from dateutil import parser
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from model_support_func import *
from model_params import *

# directories

data_dir = "/Users/Ravi/Desktop/modeling/tat_prediction/"
image_dir = "/Users/Ravi/Desktop/modeling/tat_prediction/plots/"

# load data

data = load_data(data_dir + "train_data.csv")

print("\ndata head:\n")
print(data.head())

print("\ncolumn types:\n")
print(data.dtypes)

print("#########################################\n")

# transform data

int_col = ['srno']
cat_col = ['mobmake', 'hubid', 'hubpincode', 'pickuppartnerid', 'deliverypartnerid', 'insuranceCompanyId',
           'custpincodecategory', 'claimtype', 'custpincode', 'servicecntrid']
date_col = ['memcreateddate', 'mobilepurchasedate', 'claimintdate', 'pickupStartdate']
outcome_col = 'tat_in_days'

data[outcome_col] = data[outcome_col].apply(lambda x: int(x))
data[int_col] = data[int_col].apply(lambda x: x.astype(int))
data[cat_col] = data[cat_col].apply(lambda x: x.astype('category'))
data[date_col] = data[date_col].apply(lambda x: pd.to_datetime(x, dayfirst=True, infer_datetime_format=True))

data = data.drop(['srno', 'hubpincode', 'custpincode'], axis=1)  # removing srno and duplicate column as hubid
cat_col.remove('hubpincode')
cat_col.remove('custpincode')
int_col.remove('srno')

print("\ntransformed data head:\n")
print(data.head())

print("\ncolumn types after transformation:\n")
print(data.dtypes)

print("\nnumber of data points:")
print(len(data))

print("\ndata summary:\n")
print(data.describe(include='all'))

print("\ncount of nan values:\n")
print(data.isnull().sum())

print("#########################################\n")

# feature engineering

data['diff_claim_memcrt'] = (data['claimintdate'] - data['memcreateddate']).dt.days
data['diff_claim_mobprc'] = (data['claimintdate'] - data['mobilepurchasedate']).dt.days

ref_point = pd.to_datetime('today')  # reference point can be selected as any date, for simplicity current date is taken

data['diff_ref_claim'] = (ref_point - data['claimintdate']).dt.days
data['diff_pickup_claim'] = (data['pickupStartdate'] - data['claimintdate']).dt.days

data['claim_day'] = data['claimintdate'].dt.weekday_name.astype('category')
data['pickup_day'] = data['pickupStartdate'].dt.weekday_name.astype('category')

int_col = int_col + ['diff_claim_memcrt', 'diff_claim_mobprc', 'diff_ref_claim', 'diff_pickup_claim']
cat_col = cat_col + ['claim_day', 'pickup_day']

data = data.drop(date_col, axis=1)

print("data after feature creations:\n")
print(data.head())
print("\ncolumn types:\n")
print(data.dtypes)

print("#########################################\n")

# splitting test and train data

X_train_df, y_train_df, X_test_df, y_test_df = test_train_splitter(df=data[cat_col + int_col + [outcome_col]],
                                                                   y=outcome_col,
                                                                   cat_feature_list=cat_col, int_feature_list=int_col,
                                                                   split_frac=0.8)
print("Train Data head:\n")
print(X_train_df.head())
print(y_train_df.head())
print(y_train_df.dtypes)

print("Test Data head:\n")
print(X_test_df.head())
print(y_test_df.head())
print(y_test_df.dtypes)

print("#########################################\n")
print("Train Data Length:" + str(len(X_train_df)))
print("\nTest Data Length:" + str(len(X_test_df)))
print("\n########################################\n")

print("Performing One Hot Encoding of Categorical Variables...")

print(cat_col)

X_train_labelEncoded, X_test_labelEncoded = labelEncoder_cat_features(X_train=X_train_df, X_test=X_test_df,
                                                                      cat_feature_list=cat_col)

X_train_oneHotEncoded, X_test_oneHotEncoded = oneHotEncoder_cat_features(X_train_labelEncoded=X_train_labelEncoded,
                                                                         X_test_labelEncoded=X_test_labelEncoded,
                                                                         cat_feature_list=cat_col,
                                                                         drop_last=True)

print("sample One Hot Encoded Data:\n")
print(X_train_oneHotEncoded.head())
print("\nColumn Types of One Hot Encoded Data:\n")
# print(X_test_oneHotEncoded.head())
print(X_train_oneHotEncoded.dtypes)
print(X_test_oneHotEncoded.dtypes)
# print(len(X_train_oneHotEncoded))
# print(len(X_test_oneHotEncoded))

print("#########################################\n")


###################################################################################
# Model Building
###################################################################################

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, cv=5, verbose=3,
                               random_state=42, scoring=scoring)
# Fit the random search model
rf_random.fit(X_train_oneHotEncoded, y_train_df[outcome_col])

print("best rf param:\n")

print(rf_random.best_params_)

print("#########################################\n")
print("evaluating base model:\n")
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train_oneHotEncoded, y_train_df[outcome_col])
base_accuracy = evaluate(base_model, X_test_oneHotEncoded, y_test_df[outcome_col])

print("#########################################\n")

print("evaluating best model:\n")
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test_oneHotEncoded, y_test_df[outcome_col])

print("#########################################\n")

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
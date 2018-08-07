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
from sklearn.metrics import make_scorer

# directories

data_dir = "/Users/Ravi/Desktop/modeling/tat_prediction/"
image_dir = "/Users/Ravi/Desktop/modeling/tat_prediction/plots/"

# load data

data = load_data(data_dir + "train_data.csv")

evaluation_data = load_data(data_dir + "evaluation_data.csv")

# print("\ndata head:\n")
# print(data.head())
#
# print("\ncolumn types:\n")
# print(data.dtypes)
#
# print("\ndata head:\n")
# print(evaluation_data.head())
#
# print("\ncolumn types:\n")
# print(evaluation_data.dtypes)
#
# print("#########################################\n")

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

# print("\ntransformed data head:\n")
# print(data.head())
#
# print("\ncolumn types after transformation:\n")
# print(data.dtypes)
#
# print("\nnumber of data points:")
# print(len(data))
#
# print("\ndata summary:\n")
# print(data.describe(include='all'))
#
# print("\ncount of nan values:\n")
# print(data.isnull().sum())
#
# print("#########################################\n")

# transform evaluation data

evaluation_data[int_col] = evaluation_data[int_col].apply(lambda x: x.astype(int))
evaluation_data[cat_col] = evaluation_data[cat_col].apply(lambda x: x.astype('category'))
evaluation_data[date_col] = evaluation_data[date_col].apply(lambda x: pd.to_datetime(x, dayfirst=True,
                                                                                     infer_datetime_format=True))

evaluation_data_srno = evaluation_data['srno']
evaluation_data = evaluation_data.drop(['srno', 'hubpincode', 'custpincode'], axis=1)  # removing srno and duplicate column as hubid


# print("\nsrno of evaluation data set:\n")
# print(evaluation_data_srno.head())
# print(len(evaluation_data_srno))
#
# print("\ntransformed evaluation data head:\n")
# print(evaluation_data.head())
#
# print("\ncolumn types after transformation:\n")
# print(evaluation_data.dtypes)
#
# print("\nnumber of evaluation data points:")
# print(len(evaluation_data))
#
# print("\nevaluation data summary:\n")
# print(evaluation_data.describe(include='all'))
#
# print("\ncount of nan values:\n")
# print(evaluation_data.isnull().sum())
#
# print("#########################################\n")


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

# print("data after feature creations:\n")
# print(data.head())
# print("\ncolumn types:\n")
# print(data.dtypes)
#
# print("#########################################\n")

# feature creation on evaluation data

evaluation_data['diff_claim_memcrt'] = (evaluation_data['claimintdate'] - evaluation_data['memcreateddate']).dt.days
evaluation_data['diff_claim_mobprc'] = (evaluation_data['claimintdate'] - evaluation_data['mobilepurchasedate']).dt.days

ref_point = pd.to_datetime('today')  # reference point can be selected as any date, for simplicity current date is taken

evaluation_data['diff_ref_claim'] = (ref_point - evaluation_data['claimintdate']).dt.days
evaluation_data['diff_pickup_claim'] = (evaluation_data['pickupStartdate'] - evaluation_data['claimintdate']).dt.days

evaluation_data['claim_day'] = evaluation_data['claimintdate'].dt.weekday_name.astype('category')
evaluation_data['pickup_day'] = evaluation_data['pickupStartdate'].dt.weekday_name.astype('category')

evaluation_data = evaluation_data.drop(date_col, axis=1)

# print("evaluation_data after feature creations:\n")
# print(evaluation_data.head())
# print("\ncolumn types:\n")
# print(evaluation_data.dtypes)
#
# print("#########################################\n")

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

print("Performing One Hot Encoding of Categorical Variables...\n")
print(cat_col)

X_train_labelEncoded, X_test_labelEncoded, X_eval_labelEncoded = labelEncoder_cat_features(X_train=X_train_df, X_test=X_test_df,
                                                                      X_eval=evaluation_data,
                                                                      cat_feature_list=cat_col)

X_train_oneHotEncoded, X_test_oneHotEncoded,  X_eval_oneHotEncoded = oneHotEncoder_cat_features(X_train_labelEncoded=X_train_labelEncoded,
                                                                           X_test_labelEncoded=X_test_labelEncoded,
                                                                           X_eval_labelEncoded=X_eval_labelEncoded,
                                                                           cat_feature_list=cat_col,
                                                                           drop_last=False)

# print("sample One Hot Encoded Data:\n")
# print(X_train_oneHotEncoded.head())
# print("\nColumn Types of One Hot Encoded Data:\n")
# # print(X_test_oneHotEncoded.head())
# print(X_train_oneHotEncoded.dtypes)
# print(X_test_oneHotEncoded.dtypes)
# print(X_eval_oneHotEncoded.dtypes)
# # print(len(X_train_oneHotEncoded))
# # print(len(X_test_oneHotEncoded))
#
# print("#########################################\n")


###################################################################################
# Model Building
###################################################################################

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=1,
                               random_state=42, scoring=scoring, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train_oneHotEncoded, y_train_df[outcome_col])

print("best rf param:\n")

best_param = rf_random.best_params_

print(best_param)

best_score = rf_random.best_score_

print("\nbest rf score: {:0.2f}\n".format(best_score))

results = rf_random.cv_results_
##########################################################################################################
# feature importance
##########################################################################################################
importances = rf_random.best_estimator_.feature_importances_

plt_feature_imp(importances = importances, feature_list=X_train_oneHotEncoded.columns.values, n_top_features=30,
                image_dir=image_dir)

##########################################################################################################
# hyper params plots
##########################################################################################################

hyper_params = ["n_estimators", "min_samples_split", "min_samples_leaf", "max_features", "max_depth", "bootstrap"]

for param in hyper_params:

    GridSearch_table_plot(rf_random, param_name=param, image_dir = image_dir)

##########################################################################################################

print("#########################################\n")
print("evaluating base model:\n")
base_model = RandomForestRegressor(n_estimators=10, random_state=42)
base_model.fit(X_train_oneHotEncoded, y_train_df[outcome_col])
base_accuracy = evaluate(base_model, X_test_oneHotEncoded, y_test_df[outcome_col])

print("#########################################\n")

print("evaluating best model:\n")
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test_oneHotEncoded, y_test_df[outcome_col])

print("#########################################\n")

print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

print("#########################################\n")

final_model = RandomForestRegressor(n_estimators=best_param.get("n_estimators"),
                                    min_samples_split=best_param.get("min_samples_split"),
                                    min_samples_leaf=best_param.get("min_samples_leaf"),
                                    max_features=best_param.get("max_features"),
                                    max_depth=best_param.get("max_depth"),
                                    bootstrap=best_param.get("bootstrap"),
                                    random_state=42)

final_model.fit(X_train_oneHotEncoded.append(X_test_oneHotEncoded),
                y_train_df[outcome_col].append(y_test_df[outcome_col]))

tat_pred = final_model.predict(X_eval_oneHotEncoded)

tat_pred_on_eval_data = pd.DataFrame()
tat_pred_on_eval_data['srno'] = evaluation_data_srno.values
tat_pred_on_eval_data['tat_pred'] = pd.Series(tat_pred).values

print("\npredicted tat on evaluation data:\n")
print(tat_pred_on_eval_data.head())
print("\nlength of predicted tat on evaluation data:\n")
print(len(tat_pred_on_eval_data))

tat_pred_on_eval_data.to_csv(path_or_buf=data_dir + "tat_pred_on_eval_data.csv")

print("#########################################\n")



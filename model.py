from eda_support_func import *
from dateutil import parser
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from model_support_func import *


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

data = data.drop(['srno','hubpincode'], axis=1)  # removing srno and duplicate column as hubid
cat_col.remove('hubpincode')
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

print("data after feature creations:\n")
print(data.head())
print("\ncolumn types:\n")
print(data.dtypes)

print("#########################################\n")

# splitting test and train data

X_train_df, y_train_df, X_test_df, y_test_df = test_train_splitter(df=data[cat_col+int_col+[outcome_col]], y=outcome_col,
                                                                   cat_feature_list=cat_col, int_feature_list=int_col,
                                                                   split_frac=0.8)
print("Train Data head:\n")
print(X_train_df.head())
print(y_train_df.head())

print("Test Data head:\n")
print(X_test_df.head())
print(y_test_df.head())

print("#########################################\n")
print("Train Data Length:" + str(len(X_train_df)))
print("\nTest Data Length:" + str(len(X_test_df)))
print("\n########################################\n")




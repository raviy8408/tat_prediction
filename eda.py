from eda_support_func import *
from dateutil import parser
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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

print("Column Description:\n")
print(data.describe(include=["int64"]))
print(data.describe(include=["category"]))
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


# print the categories of categorical variable
print("\nAll categorical variable levels:\n")
print_categories(data, cat_col)
print("#########################################\n")

# UVA

# outcome variable

save_hist(data=data['tat_in_days'], bins=100, alpha=0.7, image_dir=image_dir)

# histogram of integer variables

data[int_col].apply(lambda x: save_hist(data=x, bins=100, alpha=0.7, color='blue', image_dir=image_dir))

# outcome variable vs date type cols

data[date_col].apply(lambda x: save_scatter(x=x, y=data[outcome_col], image_dir=image_dir))

# outcome variables vs categorical data

save_boxplots_for_cat_var(data=data, cat_var_list=cat_col, y=outcome_col, image_dir=image_dir)
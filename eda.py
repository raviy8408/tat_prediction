from support_func import *
from dateutil import parser
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

int_col = ['srno', 'tat_in_days']

cat_col = ['mobmake', 'hubid', 'hubpincode', 'pickuppartnerid', 'deliverypartnerid', 'insuranceCompanyId',
           'custpincodecategory', 'claimtype', 'custpincode', 'servicecntrid']

date_col = ['memcreateddate', 'mobilepurchasedate', 'claimintdate', 'pickupStartdate']

outcome_col = 'tat_in_days'

data[int_col] = data[int_col].apply(lambda x: x.astype(int))
data[cat_col] = data[cat_col].apply(lambda x: x.astype('category'))
data[date_col] = data[date_col].apply(lambda x: x.apply(parser.parse))

data = data.drop(['hubpincode'], axis= 1) # duplicate column as hubid
cat_col.remove('hubpincode')

print("\ntransformed data head:\n")
print(data.head())

print("\ncolumn types after transformation:\n")
print(data.dtypes)

print("\nnumber of data points:")
print(len(data))

print("\ndata summary:\n")
print(data.describe(include= 'all'))

print("\ncount of nan values:\n")
print(data.isnull().sum())

print("#########################################\n")


# UVA

# outcome variable

save_hist(data=data['tat_in_days'],bins=100,alpha=0.7,image_dir=image_dir)

# outcome variable vs date type cols

data[date_col].apply(lambda x: save_scatter(x= x, y= data[outcome_col], image_dir= image_dir))

# outcome variables vs categorical data

save_boxPlots_for_cat_var(data= data, cat_var_list= cat_col, y = outcome_col, image_dir= image_dir)
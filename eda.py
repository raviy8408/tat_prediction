from support_func import *
from dateutil import parser

# load data

data = load_data("/Users/Ravi/Desktop/modeling/tat_prediction/train_data.csv")

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

data[int_col] = data[int_col].apply(lambda x: x.astype(int))

data[cat_col] = data[cat_col].apply(lambda x: x.astype('category'))

data[date_col] = data[date_col].apply(lambda x: x.apply(parser.parse))

print("\ncolumn types after transformation:\n")
print(data.dtypes)

print("#########################################\n")


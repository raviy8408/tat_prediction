import pandas as pd
import numpy as np


def test_train_splitter(df, y, cat_feature_list, int_feature_list, outcome_type='int', split_frac=0.8):
    '''
    Splits the data into test and train in the ration provided and returns 4 data frames:
    x_train, y_train, x_test, y_test
    :param df: complete data
    :param y: outcome variable
    :param cat_feature_list: all the categorical variables
    :return: x_train, y_train, x_test, y_test
    '''
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df.drop([y], axis=1).as_matrix(),
                                                        df[y].as_matrix(), test_size=1 - split_frac)

    # test train dataframe creation
    X_train_df = pd.DataFrame(data=X_train, columns=df.drop([y], axis=1).columns.values)
    y_train_df = pd.DataFrame(data=y_train, columns=[y])

    X_test_df = pd.DataFrame(data=X_test, columns=df.drop([y], axis=1).columns.values)
    y_test_df = pd.DataFrame(data=y_test, columns=[y])

    # assigning variable types to test and train data columns
    X_train_df[cat_feature_list] = X_train_df[cat_feature_list].apply(lambda x: x.astype('category'))
    X_test_df[cat_feature_list] = X_test_df[cat_feature_list].apply(lambda x: x.astype('category'))

    X_train_df[int_feature_list] = X_train_df[int_feature_list].apply(lambda x: x.astype('int64'))
    X_test_df[int_feature_list] = X_test_df[int_feature_list].apply(lambda x: x.astype('int64'))

    _non_float_feature_list = cat_feature_list + [y] + int_feature_list

    X_train_df[df.columns.difference(_non_float_feature_list)] = X_train_df[
        df.columns.difference(_non_float_feature_list)].apply(lambda x: x.astype('float'))
    X_test_df[df.columns.difference(_non_float_feature_list)] = X_test_df[
        df.columns.difference(_non_float_feature_list)].apply(lambda x: x.astype('float'))

    y_train_df[[y]] = y_train_df[[y]].apply(lambda x: x.astype(outcome_type))
    y_test_df[[y]] = y_test_df[[y]].apply(lambda x: x.astype(outcome_type))

    return X_train_df, y_train_df, X_test_df, y_test_df


def labelEncoder_cat_features(X_train, X_test, cat_feature_list):
    '''
    Converts all categorical features to numerical levels
    :param X_train: train data frame
    :param X_test: test data frame
    :param cat_feature_list: all categorical variables
    :return: encoded pandas data frame
    '''
    from sklearn.preprocessing import LabelEncoder

    X_train[cat_feature_list] = X_train[cat_feature_list].apply(lambda x: x.astype(str))
    X_test[cat_feature_list] = X_test[cat_feature_list].apply(lambda x: x.astype(str))

    le = LabelEncoder()
    # Iterating over all the common columns in train and test
    for col in X_test.columns.values:
        # Encoding only categorical variables
        if X_test[col].dtypes == 'object':
            # Using whole data to form an exhaustive list of levels
            data = X_train[col].append(X_test[col])
            le.fit(data.values)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

    return X_train, X_test


def oneHotEncoder_cat_features(X_train_labelEncoded, X_test_labelEncoded, cat_feature_list, drop_last=False):
    '''
    creates one hot encoded data frame
    :param X_train_labelEncoded: label encoded train data frame
    :param X_test_labelEncoded: label encoded test data frame
    :param cat_feature_list: all the categorical features
    :return:
    '''

    from sklearn.preprocessing import OneHotEncoder

    X_train_oneHotEncoded = X_train_labelEncoded
    X_test_oneHotEncoded = X_test_labelEncoded

    enc = OneHotEncoder(sparse=False)

    for col in cat_feature_list:
        data = X_train_labelEncoded[[col]].append(X_test_labelEncoded[[col]])
        enc.fit(data)
        # Fitting One Hot Encoding on train data
        temp = enc.transform(X_train_labelEncoded[[col]])
        # Changing the encoded features into a data frame with new column names
        # if number of categorical levels is less than 3, keep only one column
        if drop_last == False:
            temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col]
                                .value_counts().index])
        elif drop_last == True:
            temp = pd.DataFrame(temp[:, :-1], columns=[(col + "_" + str(i)) for i in data[col]
                                .value_counts().index[:-1]])
        # In side by side concatenation index values should be same
        # Setting the index values similar to the X_train data frame
        temp = temp.set_index(X_train_labelEncoded.index.values)
        # adding the new One Hot Encoded varibales to the train data frame
        X_train_oneHotEncoded = pd.concat([X_train_oneHotEncoded, temp], axis=1)
        # fitting One Hot Encoding on test data
        temp = enc.transform(X_test_labelEncoded[[col]])
        # changing it into data frame and adding column names
        # if number of categorical levels is less than 3, keep only one column
        if drop_last == False:
            temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col]
                                .value_counts().index])
        elif drop_last == True:
            temp = pd.DataFrame(temp[:, :-1], columns=[(col + "_" + str(i)) for i in data[col]
                                .value_counts().index[:-1]])
        # Setting the index for proper concatenation
        temp = temp.set_index(X_test_labelEncoded.index.values)
        # adding the new One Hot Encoded varibales to test data frame
        X_test_oneHotEncoded = pd.concat([X_test_oneHotEncoded, temp], axis=1)

    # dropping original categorical columns
    X_train_oneHotEncoded = X_train_oneHotEncoded.drop(cat_feature_list, axis=1)
    X_test_oneHotEncoded = X_test_oneHotEncoded.drop(cat_feature_list, axis=1)

    return X_train_oneHotEncoded, X_test_oneHotEncoded


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    # print(predictions)
    # print(test_labels.values)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.2f} days.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(data, col_target, col_time, train_size=0.4, valid_size=0.3, test_size=0.3, random_state=42):
    df_temp = data.loc[:,[col_time, col_target]]
    df_temp.loc[:,'stratify'] = data.loc[:,col_time].astype('str')+"_"+data.loc[:,col_target].astype('str')
    X_train, X_test, y_train, y_test = train_test_split(df_temp['stratify'], df_temp[col_target], test_size=test_size,
                                                        random_state=random_state, stratify=df_temp['stratify'])
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size/(train_size+valid_size),
                                                        random_state=random_state, stratify=X_train)


    df_temp.loc[X_train.index,"data_type"] = "train"
    df_temp.loc[X_valid.index, "data_type"] = "valid"
    df_temp.loc[X_test.index, "data_type"] = "test"
    return df_temp.loc[:,"data_type"]


if __name__ == "__main__":
    pass
    # pd.set_option('display.max_columns',100)
    # pd.set_option('display.max_rows', 100)
    # path_data = 'H:\\Datascience\\Data\\Telecom_customer churn.csv'
    #
    # col_time = 'col_time'
    # col_target = 'churn'
    # col_id = 'Customer_ID'
    #
    # data = pd.read_csv(path_data)
    # data['col_time'] = pd.qcut(data['Customer_ID'], 10, labels=False)
    # data['data_type'] = split_data(data, col_target, col_time, train_size=0.4, valid_size=0.3, test_size=0.3)
    # print(data[['data_type', col_target, col_time]].groupby(['data_type', col_time]).mean())
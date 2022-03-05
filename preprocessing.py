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

def which_is_missing(data, col_cat, col_target, train_index, test_index, valid_index=None, min_size=0, print_summary=False):
    df_cat = data[[col_cat, col_target, 'data_type']]
    unique_overall = set(df_cat[col_cat].unique())
    df_train = pd.DataFrame(columns=unique_overall)
    df_target = df_train.copy()
    df_test = df_train.copy()
    if valid_index is not None: df_valid = df_train.copy()
    sr_count_missing = pd.Series(index=unique_overall, data=np.zeros(len(unique_overall)))

    unique_train = set(df_cat.loc[train_index, col_cat].unique())
    unique_test = set(df_cat.loc[test_index, col_cat].unique())
    if valid_index is not None:
        unique_valid = set(df_cat.loc[valid_index, col_cat].unique())
        sym_dif_1 = unique_train.symmetric_difference(unique_test)
        sym_dif_2 = unique_train.symmetric_difference(unique_valid)
        sym_dif_3 = unique_valid.symmetric_difference(unique_test)
        sym_dif = sym_dif_1.union(sym_dif_2).union(sym_dif_3)
    else:
        sym_dif = unique_train.symmetric_difference(unique_test)
    sr_count_missing[sym_dif] += 1
    sr_count_missing.name='missing'
    df_train = df_train.append(df_cat.loc[train_index, col_cat].value_counts(), ignore_index=True).transpose()
    df_train.columns = ['count_train']
    if valid_index is not None:
        df_valid = df_valid.append(df_cat.loc[valid_index, col_cat].value_counts(), ignore_index=True).transpose()
        df_valid.columns = ['count_valid']
    df_test = df_test.append(df_cat.loc[test_index, col_cat].value_counts(), ignore_index=True).transpose()
    df_test.columns = ['count_test']
    df_target = df_target.append(df_cat.loc[train_index, [col_cat, col_target]].groupby([col_cat]).mean().transpose(), ignore_index=True).transpose()
    # else:
    #     df_target = df_target.append(
    #         df_cat.loc[train_index|valid_index, [col_cat, col_target]].groupby([col_cat]).mean().transpose(),
    #         ignore_index=True).transpose()
    df_target.columns = ['target_mean']

    if valid_index is not None:
        df_summary = df_train.join(df_test, how='left'). \
            join(df_valid, how='left'). \
            join(pd.DataFrame(sr_count_missing, columns=['missing']), how='left'). \
            join(df_target, how='left')
        df_summary['min_count'] = df_summary[['count_train', 'count_valid', 'count_test']].min(axis=1)
    else:
        df_summary = df_train.join(df_test, how='left'). \
            join(pd.DataFrame(sr_count_missing, columns=['missing']), how='left'). \
            join(df_target, how='left')
        df_summary['min_count'] = df_summary[['count_train', 'count_test']].min(axis=1)

    #
    print(f"Predictor: {col_cat}")
    print("==================================================================")
    print(f"List of predictors: {df_summary.index.tolist()}")
    print(f"Missing categories: {df_summary.loc[df_summary['missing']>0].sort_values(['missing'], ascending=False).index.tolist()}")
    print(f"Categories <{min_size} samples: {df_summary.loc[df_summary['min_count']<min_size].sort_values(['min_count'], ascending=True).index.tolist()}")
    print(f"Consider to merge this categories: {set(df_summary.loc[df_summary['missing']>0].sort_values(['missing'], ascending=False).index.tolist()).union(set(df_summary.loc[df_summary['min_count']<min_size].sort_values(['min_count'], ascending=True).index.tolist()))}")
    if print_summary: print(df_summary.sort_values(by=['missing', 'min_count'], ascending=[False, True]))
    print("")

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
    # col_number = data.select_dtypes(include='number').columns.tolist()
    # col_number.remove(col_id)
    # col_number.remove(col_target)
    # col_object = data.select_dtypes(include='object').columns.tolist()
    # data['col_time'] = pd.qcut(data['Customer_ID'], 10, labels=False)
    # data['data_type'] = split_data(data, col_target, col_time, train_size=0.4, valid_size=0.3, test_size=0.3)
    # # print(data[['data_type', col_target, col_time]].groupby(['data_type', col_time]).mean())
    # which_is_missing(data, col_object[0], col_target, (data['data_type']=='train'), (data['data_type']=='test'), min_size=100, h_mul=1, v_mul=1)
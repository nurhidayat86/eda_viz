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

def which_is_missing(data, col_cat, col_target, train_index, test_index, valid_index=None, min_size=0, print_summary=False, merge_type='mode'):
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

    missing_category = df_summary.loc[df_summary['missing']>0].sort_values(['missing'], ascending=False).index.tolist()
    min_samples = df_summary.loc[df_summary['min_count']<min_size].sort_values(['min_count'], ascending=True).index.tolist()
    mode_sample = df_summary.sort_values(['min_count'], ascending=False).index.tolist()[0]
    least_sample = df_summary.sort_values(['min_count'], ascending=True).index.tolist()[1]
    merge_categories = list(set(missing_category).union(set(min_samples)))

    if len(merge_categories) == 1 and merge_type == 'mode':
        possible_merge = merge_categories + [mode_sample]
    elif len(merge_categories) == 1 and merge_type == 'least':
        possible_merge = merge_categories + [least_sample]
    else:
        possible_merge = merge_categories

    print(f"Predictor: {col_cat}")
    print("==================================================================")
    print(f"List of predictors: {df_summary.index.tolist()}")
    print(f"Missing categories: {missing_category}")
    print(f"Categories <{min_size} samples: {min_samples}")
    print(f"Consider to merge this categories: {possible_merge}")
    if print_summary: print(df_summary.sort_values(by=['missing', 'min_count'], ascending=[False, True]))
    print("")
    output_dict = df_summary.to_dict()
    output_dict['predictor'] = col_cat
    output_dict['categories'] = df_summary.index.tolist()
    output_dict['Missing categories'] = missing_category
    output_dict['min samples'] = min_samples
    output_dict['possible merge'] = possible_merge
    output_dict['mode sample'] = mode_sample
    output_dict['least sample'] = least_sample
    return output_dict

def rare_category_grouper(data, cols_cat, list_dict, replace=False, col_name='_rare', var_name='_r1'):
    cols_cat_rare = cols_cat.copy()
    df_temp = pd.DataFrame(index=data.index)
    for i_dict in list_dict:
        indexer = i_dict['predictor']
        missing_cat = i_dict['Missing categories']
        possible_merge = i_dict["possible merge"]
        if len(possible_merge) > 0:
            print(f"Predictor {indexer}: {missing_cat} --> {possible_merge}")
            cols_cat_rare.remove(indexer)
            cols_cat_rare.append(indexer+col_name)
            df_temp[indexer+col_name] = data[indexer].values
            for i_merge in possible_merge:
                df_temp.loc[df_temp[indexer+col_name] == i_merge, indexer+col_name] = indexer+var_name
    return cols_cat_rare, df_temp



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
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.ticker import PercentFormatter

def nan_mat(data, col_feature, time, threshold=0):
    temp = data[col_feature + [time]].groupby(time).apply(
        lambda x: x.isna().sum() / (x.count() + x.isna().sum()))
    temp = temp.drop([time], axis=1)
    temp = temp.transpose()
    temp['avg'] = temp.mean(axis=1)
    temp = temp.sort_values(by=['avg'], ascending=False)
    temp = temp.loc[temp['avg']>threshold,:]
    an_matrix = (100*temp.values).astype('int')
    plt.figure(figsize=(1*len(temp.columns),0.5*len(temp)))
    plt.imshow(temp, vmin=0, vmax=1, interpolation='nearest', cmap='Blues')
    for i in range(len(temp)):
        for j in range(len(temp.columns)):
            text = plt.text(j, i, f"{an_matrix[i, j]}",
                           ha="center", va="center", color="black")
    plt.xticks(np.arange(0, len(temp.columns)), temp.columns, rotation=45, ha="right", rotation_mode="anchor")
    plt.xlabel('Predictors')
    plt.yticks(np.arange(0, len(temp.index)), temp.index)
    plt.xlabel(f'Time ({time})')
    plt.title('% NAN per predictor')
    plt.tight_layout()
    plt.show()
    plt.close()

def nan_share(data, col_feature, time, threshold=0):
    temp = data[col_feature + [time]].groupby(time).apply(lambda x: x.isna().sum() / (x.count() + x.isna().sum()))
    temp = temp.drop([time], axis=1)
    temp = temp.transpose()
    temp['avg'] = temp.mean(axis=1)
    temp = temp.loc[temp['avg']>threshold,:]
    temp_avg = temp['avg'].copy()
    temp = temp.drop(['avg'], axis=1)
    temp = temp.transpose()
    col_feature = temp.columns.tolist()
    n = len(col_feature)
    # print(temp)
    # print(temp_avg)
    for i in range(0, n):
        arr_avg = [temp_avg[col_feature[i]] for j in range(0, len(temp.index))]
        print(arr_avg)
        plt.figure()
        plt.ylim(-0.1, 1)
        plt.title(col_feature[i])
        plt.ylabel('Nan share')
        plt.xlabel(f'Time in {time}')
        plt.plot([f"{i}" for i in temp.index], temp[col_feature[i]])
        plt.plot([f"{i}" for i in temp.index], arr_avg, linestyle='dashed', color='r')
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        # plt.xticks(np.arange(temp.index[0], temp.index[-1]+1,step_xticks), rotation='vertical')
        plt.tight_layout()
        plt.show()
        plt.close()

def histogram(data, col_feature, n_bins=50):
    for i in range(0, len(col_feature)):
        temp=data[col_feature[i]]
        fig, ax = plt.subplots(tight_layout=True)
        ax.hist(temp, bins=n_bins)
        ax.set_title(col_feature[i])
        plt.show()
        plt.close(fig)

def num_stability(data, col_feature, time, n_bins=5, bar_width=0.75, w_mul=1, h_mul=0.5, enable_anottation=True):
    for i in range(0, len(col_feature)):
        temp = data.loc[:,[col_feature[i], time]]
        nan_val = temp[col_feature[i]].abs().max()*(-2)
        len_feature = len(temp[col_feature[i]])
        temp[col_feature[i]] = temp[col_feature[i]].fillna(nan_val)
        temp['binning'], bins = pd.qcut(temp[col_feature[i]], n_bins, retbins=True, duplicates='drop')

        df_average = temp[[col_feature[i], 'binning']].groupby(['binning']).count()/len_feature
        df_len = temp[[col_feature[i], time]].groupby([time]).count()
        df_time = temp[[col_feature[i], time, 'binning']].groupby([time, 'binning']).count()
        first = df_time.index.get_level_values(time)
        second = df_time.index.get_level_values('binning')
        df_time['len'] = df_len.loc[first].values
        df_time['share'] = df_time[col_feature[i]].divide(df_time['len'])
        df_time['overall_share'] = df_average.loc[second].values
        df_time['diff'] = (df_time['overall_share']-df_time['share']).abs()
        df_time.fillna(0, inplace=True)
        df_time = df_time.reset_index()
        # print(df_average)

        labels = ['base'] + df_len.index.astype('str').tolist()
        w_size = w_mul*(len(labels))
        h_size = h_mul*(len(df_average.index)+1)
        if h_size <= 3: h_size=3
        fig, ax = plt.subplots(figsize=(w_size, h_size))
        bottom_bin = [0 for i in labels]

        for idx_threshold in df_average.index.tolist():
            temp_values = df_average.loc[idx_threshold].values.tolist() + df_time.loc[df_time['binning'] == idx_threshold, 'share'].values.tolist()
            rects = ax.bar(labels, temp_values, width=bar_width, bottom=bottom_bin, label=idx_threshold)
            bottom_bin = [bottom_bin[ijk] + temp_values[ijk] for ijk in range(len(bottom_bin))]

            if enable_anottation:
                for p in rects.patches:
                    width, height = p.get_width(), p.get_height()
                    x, y = p.get_xy()
                    ax.text(x + width / 2,
                            y + height / 2,
                            '{:.0f}%'.format(height*100),
                            horizontalalignment='center',
                            verticalalignment='center')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xticks(np.arange(0, len(labels)))
        ax.set_xticklabels(labels, rotation=45, rotation_mode="anchor", ha="right")
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f"{i_label}%" for i_label in np.arange(0,110,10)])
        ax.set_ylabel('Share')
        ax.set_title(f"{col_feature[i]}, NaN val: {nan_val}")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

def corr_mat(data, col_feature, col_target, time, method='spearman', threshold=0.05, h_mul=0.4, v_mul=0.4, transpose=False):
    temp = data[col_feature + [col_target, time]].groupby(time).apply(
        lambda x: x[col_feature].corrwith(x[col_target], method=method)
    )
    temp = temp.transpose()
    temp['avg'] = temp.mean(axis=1)
    temp['sort'] = temp['avg'].abs()
    temp = temp.sort_values(by=['sort'], ascending=False)
    temp = temp.loc[temp['sort'] > threshold,:]
    temp = temp.drop(['sort'], axis=1)
    if transpose:
        temp = temp.transpose()
    an_matrix = (100*temp.values).astype('int')
    if transpose:
        h_size = v_mul*(len(temp.columns)+0.5)
        v_size = h_mul*(len(temp)+1.25)
    else:
        h_size = h_mul*(len(temp.columns)+1.25)
        v_size = v_mul*(len(temp)+0.5)
    plt.figure(figsize=(h_size, v_size))
    # plt.figure()
    cax = plt.imshow(temp, vmin=-1, vmax=1, interpolation='nearest', cmap='seismic')
    for i in range(len(temp)):
        for j in range(len(temp.columns)):
            text = plt.text(j, i, f"{an_matrix[i, j]}",
                           ha="center", va="center", color="black")
    plt.xticks(np.arange(0, len(temp.columns)), temp.columns, rotation=45, ha="right", rotation_mode="anchor")
    # cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
    # cbar.ax.set_yticklabels(['-100%', '0%', '100%'])
    plt.xlabel('Predictors')
    plt.yticks(np.arange(0, len(temp.index)), temp.index)
    if transpose:
        plt.ylabel(f'Time ({time})')
        plt.xlabel(f'Predictor')
    else:
        plt.xlabel(f'Time ({time})')
        plt.ylabel(f'Predictor')
    plt.title(f'% Correlation ({method}) with {col_target}')
    plt.tight_layout()
    plt.show()
    plt.close()

def perf_mat(data, col_feature, col_target, time, method='gini', threshold=0.01, h_mul=0.4, v_mul=0.5, transpose=False):
    for i in range(0, len(col_feature)):
        df_temp = data.loc[data[col_feature[i]].isnull()==False,[col_feature[i],col_target, time]].groupby(time).apply(
                lambda x: roc_auc_score(x[col_target], x[col_feature[i]])
        )

        df_temp = df_temp.to_frame(name=col_feature[i])

        if i==0:
            temp = df_temp
        else:
            # print(df_temp)
            temp = temp.join(df_temp, how='left')

    temp = temp.transpose()
    temp['avg'] = temp.mean(axis=1)

    if method == 'gini':
        temp = 2*temp-1

    temp['sort'] = temp['avg'].abs()
    temp = temp.sort_values(by=['sort'], ascending=False)
    temp = temp.loc[temp['sort'] > threshold,:]
    temp = temp.drop(['sort'], axis=1)
    # print(temp)

    if transpose == True:
        temp = temp.transpose()
        h_size = h_mul*(len(temp.columns)+1)
        v_size = v_mul*(len(temp)+1)+1
    else:
        h_size = v_mul*(len(temp.columns)+1)+1
        v_size = h_mul*(len(temp)+1)

    an_matrix = (100 * temp.values).round(1)
    # print(f"{h_size}:{v_size}")
    plt.figure(figsize=(h_size, v_size))
    # plt.figure()
    cax = plt.imshow(temp, vmin=-1, vmax=1, interpolation='nearest', cmap='seismic')
    for i in range(len(temp)):
        for j in range(len(temp.columns)):
            text = plt.text(j, i, f"{an_matrix[i, j]}",
                           ha="center", va="center", color="black")
    plt.xticks(np.arange(0, len(temp.columns)), temp.columns, rotation=45, ha="right", rotation_mode="anchor")
    # cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
    # cbar.ax.set_yticklabels(['-100%', '0%', '100%'])
    plt.yticks(np.arange(0, len(temp.index)), temp.index)
    if transpose:
        plt.ylabel(f'Time ({time})')
        plt.xlabel('Predictors')
    else:
        plt.xlabel(f'Time ({time})')
        plt.ylabel('Predictors')
    plt.title(f'% {method} predictors to predict {col_target}')
    plt.tight_layout()
    plt.show()
    plt.close()

def is_granular(data, col_cat, col_target, n_splits=100, test_size=0.4, random_state=1234, min_size=100, MAE=0.3, h_mul=1, v_mul=1):
    df_cat = data[[col_cat, col_target]]
    unique_overall = set(df_cat[col_cat].unique())
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    sss.get_n_splits(df_cat[col_cat], df_cat[col_target])
    df_train = pd.DataFrame(columns=unique_overall)
    df_target = df_train.copy()
    df_test = df_train.copy()
    sr_count_missing = pd.Series(index=unique_overall, data=np.zeros(len(unique_overall)))

    for train_index, test_index in sss.split(df_cat[col_cat], df_cat[col_target]):
        unique_train = set(df_cat.loc[train_index, col_cat].unique())
        unique_test = set(df_cat.loc[test_index, col_cat].unique())
        sym_dif = unique_train.symmetric_difference(unique_test)
        sr_count_missing[sym_dif] += 1
        df_train = df_train.append(df_cat.loc[train_index, col_cat].value_counts(), ignore_index=True)
        df_test = df_test.append(df_cat.loc[test_index, col_cat].value_counts(), ignore_index=True)
        df_target = df_target.append(df_cat.loc[train_index, [col_cat, col_target]].groupby([col_cat]).mean().transpose(), ignore_index=True)

    df_train = df_train.fillna(0)
    df_train = df_train.agg(['min','mean','std', 'max']).transpose()
    df_train.columns = df_train.columns + '_count_of_train'
    df_test = df_test.fillna(0)
    df_test = df_test.agg(['min', 'mean', 'std', 'max']).transpose()
    df_test.columns = df_test.columns + '_count_of_test'

    df_target = df_target.loc[:,unique_overall].agg(['mean', 'std']).transpose()
    df_target['MAE'] = (df_target['std'].divide(df_target['mean'])).abs()
    df_target['mean-std'] = df_target['mean']  - df_target['std']
    df_target['mean+std'] = df_target['mean'] + df_target['std']
    df_target.columns = df_target.columns + '_target'

    df_summary = df_train.join(df_test, how='left').\
        join(pd.DataFrame(sr_count_missing, columns=['count_missing']),how='left').\
        join(df_target, how='left')

    df_summary['min_samples'] = df_summary[['min_count_of_train', 'min_count_of_test']].min(axis=1)

    # print(df_summary.loc[(df_summary.min_train>=min_size)&(df_summary.min_test>=min_size)&(df_summary.count_missing==0),:])
    # print(df_summary.sort_values(['min_count_of_test','count_missing'], ascending=[True, False]))

    print(f"Predictor: {col_cat}")
    print("==================================================================")
    print(f"List of predictors: {df_summary.index.tolist()}")
    print(f"Possible missing categories: {df_summary.loc[df_summary['count_missing']>0].sort_values(['count_missing'], ascending=False).index.tolist()}")
    print(f"Categories <{min_size} samples: {df_summary.loc[df_summary['min_samples']<min_size].sort_values(['min_samples'], ascending=True).index.tolist()}")
    print(f"Categories with target mean variance > {MAE}: {df_summary.loc[df_summary['MAE_target'] > MAE].sort_values(['MAE_target'], ascending=False).index.tolist()}")
    print(f"Consider to merge this categories: {set(df_summary.loc[df_summary['count_missing']>0].sort_values(['count_missing'], ascending=False).index.tolist()).union(set(df_summary.loc[df_summary['min_samples']<min_size].sort_values(['min_samples'], ascending=True).index.tolist())).union(set(df_summary.loc[df_summary['MAE_target'] > MAE].sort_values(['MAE_target'], ascending=False).index.tolist()))}")
    print("")

    # fig, ax = plt.subplots(figsize=(h_mul*df_summary.shape[0], v_mul*df_summary.shape[1]))
    # # hide axes
    # fig.patch.set_visible(False)
    # ax.axis('off')
    # ax.axis('tight')
    # ax.table(cellText=df_summary.values, colLabels=df_summary.columns, loc='center')
    # fig.tight_layout()
    # plt.show()
    # plt.close(fig)

def target_mean(data, col_target, time):
    df_summary = data[[col_target, time]].groupby(time).agg(['mean', 'count'])
    # print(df_summary['churn'])
    fig, ax = plt.subplots()
    ax.bar(df_summary.index, df_summary[(col_target,'count')], color='C0')
    ax2 = ax.twinx()
    ax2.plot(df_summary.index, (10*df_summary[(col_target,'mean')]).round(2), color='C1')
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    ax.set_xticks(df_summary.index.tolist())
    ax.set_xticklabels(df_summary.index.tolist(), rotation=45, ha='right')
    ax.set_ylabel('count data', color='C0')
    ax2.set_ylabel('% target mean', color='C1')
    ax.set_xlabel(time)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def cat_stability(data, col_feature, time, bar_width=0.75, w_mul=1, h_mul=0.5, enable_anottation=True):
    for i in range(0, len(col_feature)):
        temp = data.loc[:,[col_feature[i], time]]
        nan_val = 'NA'
        len_feature = len(temp[col_feature[i]])
        temp[col_feature[i]] = temp[[col_feature[i]]].fillna(nan_val)
        temp['binning'] = temp[col_feature[i]]
        df_average = temp[[col_feature[i], 'binning']].groupby('binning').count()/len_feature
        print(df_average)
        df_len = temp[[col_feature[i], time]].groupby([time]).count()
        df_time = temp[[col_feature[i], time, 'binning']].groupby([time, 'binning']).count()
        first = df_time.index.get_level_values(time)
        second = df_time.index.get_level_values('binning')
        df_time['len'] = df_len.loc[first].values
        df_time['share'] = df_time[col_feature[i]].divide(df_time['len'])
        df_time['overall_share'] = df_average.loc[second].values
        df_time['diff'] = (df_time['overall_share']-df_time['share']).abs()
        df_time.fillna(0, inplace=True)
        df_time = df_time.reset_index()
        # print(df_average)

        labels = ['base'] + df_len.index.astype('str').tolist()
        w_size = w_mul*(len(labels))
        h_size = h_mul*(len(df_average.index)+1)
        if h_size <= 3: h_size=3
        fig, ax = plt.subplots(figsize=(w_size, h_size))
        bottom_bin = [0 for i in labels]

        for idx_threshold in df_average.index.tolist():
            temp_values = df_average.loc[idx_threshold].values.tolist() + df_time.loc[df_time['binning'] == idx_threshold, 'share'].values.tolist()
            rects = ax.bar(labels, temp_values, width=bar_width, bottom=bottom_bin, label=idx_threshold)
            bottom_bin = [bottom_bin[ijk] + temp_values[ijk] for ijk in range(len(bottom_bin))]

            if enable_anottation:
                for p in rects.patches:
                    width, height = p.get_width(), p.get_height()
                    x, y = p.get_xy()
                    ax.text(x + width / 2,
                            y + height / 2,
                            '{:.0f}%'.format(height*100),
                            horizontalalignment='center',
                            verticalalignment='center')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xticks(np.arange(0, len(labels)))
        ax.set_xticklabels(labels, rotation=45, rotation_mode="anchor", ha="right")
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f"{i_label}%" for i_label in np.arange(0,110,10)])
        ax.set_ylabel('Share')
        ax.set_title(f"{col_feature[i]}, NaN val: {nan_val}")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    # pass
    pd.set_option('display.max_columns',100)
    pd.set_option('display.max_rows', 100)
    path_data = 'H:\\Datascience\\Data\\Telecom_customer churn.csv'

    col_time = 'col_time'
    col_target = 'churn'
    col_id = 'Customer_ID'

    data = pd.read_csv(path_data)
    col_number = data.select_dtypes(include='number').columns.tolist()
    col_number.remove(col_id)
    col_number.remove(col_target)
    col_object = data.select_dtypes(include='object').columns.tolist()
    data['col_time'] = pd.qcut(data['Customer_ID'], 10, labels=False)
    # print(data[col_object].columns)
    # # corrmat = data[col_number].corrwith(data[col_target], method='spearman')
    # # corr_mat(data, col_number, col_target, col_time, method='spearman', transpose=False)
    # # print(corrmat)
    # # num_stability(data, col_number, col_time, n_bins=5)
    # perf_mat(data, col_number, col_target, col_time, method='gini', threshold=0.05, transpose=True)
    # is_granular(data, col_object[1], col_target, n_splits=100, test_size=0.4, random_state=1234, min_size=100)
    # target_mean(data, col_target, col_time)
    cat_stability(data, col_object, col_time, bar_width=0.75, w_mul=1, h_mul=0.5, enable_anottation=True)
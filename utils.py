import pandas as pd
import numpy as np
from itertools import chain
import itertools
import math
import matplotlib.pyplot as plt
input_path = 'inputs/'
output_path = 'outputs/'
####data####
Performance_12mob = 'performance_16mob.csv'
Performance_shape = 'performance_20mob.csv'
####seg####
segmentations = pd.read_csv(input_path + 'rankings.csv').drop(columns=['if_closed', 'risk_ranking', 'rev_ranking'])
seg = segmentations.columns.to_list()
risk_seg = 'SEGMENTATION_VAR_4'
rev_seg = 'SEGMENTATION_VAR_5'
####rankings###
rankings = pd.read_csv(input_path + 'rankings.csv', index_col= list(range(0,len(seg))))
risk_order = rankings.reset_index()[[risk_seg, 'risk_ranking']].drop_duplicates()[risk_seg].to_list()
rev_order = rankings.reset_index()[[rev_seg, 'rev_ranking']].drop_duplicates()[rev_seg].to_list()



#########helper functions for data preprocessing########

def cleaning(df):
    lst = list(df.select_dtypes(['object']).columns)
    exclude = seg + ['ISSUE_DT', 'MOB']
    objects = [c for c in lst if c not in exclude]
    df[objects] = df[objects].apply(pd.to_numeric)
    return df

def pivot_out_full(df, var, seg):
    seg = seg + ['MOB']
    return df.groupby(seg)[var].sum().unstack('MOB').fillna(0)

def metrics_cal(piv_dict):
    piv_dict['ulr_per_open'] = piv_dict['mthly_writeoffs']/piv_dict['open_accts']
    piv_dict['bal_util'] = piv_dict['balance']/piv_dict['credit_limit']
    piv_dict['pur_util'] = piv_dict['spend']/piv_dict['credit_limit']
    piv_dict['revol_util'] = piv_dict['revolve_balance']/piv_dict['balance']
    
    piv_dict['lmt_per_open'] = piv_dict['credit_limit']/piv_dict['open_accts']
    piv_dict['open_per_booked'] = piv_dict['open_accts']/piv_dict['booked_accts']
    piv_dict['annual_fee'] = piv_dict['annual_fee']/piv_dict['open_accts']
    return piv_dict


def aggregate(df, var, seg):
    piv_dict = {}
    for i in var:
        df1 = pivot_out_agg_sample(df, i, seg)
        piv_dict[i] = df1
    return piv_dict       

def pivot_out_agg_sample(df, var, seg):
    return df[df['MOB']==1].groupby(seg)[var].sum().fillna(0).to_frame().rename(columns={var:'valuesum'})


def pivot_out_seg(df, var, seg):
    return df.groupby(seg)[var].sum().unstack('MOB').fillna(0)
    
    
def df_preprocess(df, var, seg_lst):
    piv_dict={}
    for i in var:
        df1 = pivot_out_seg(df, i , seg_lst)
        piv_dict[i] = df1
    return piv_dict

def view_all_smooth_method(shape, curve, mob_start, mob_end):
    fig_size = (20,10)
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize = fig_size, dpi = 180)
    print(shape.index.values[curve-1])
    plot_curve_sbs_method(shape.iloc[curve-1:curve,:],smooth_to_average_mob(shape, curve, mob_start, mob_end).iloc[curve-1:curve,:], 'smooth to average level', 1)
    plot_curve_sbs_method(shape.iloc[curve-1:curve,:],smooth_to_ma_mob(shape, curve, mob_start, mob_end,3).iloc[curve-1:curve,:], 'smooth to moving average', 2)
    plot_curve_sbs_method(shape.iloc[curve-1:curve,:],smooth_to_max_mob(shape, curve, mob_start, mob_end).iloc[curve-1:curve,:], 'smooth to max', 3)
    plot_curve_sbs_method(shape.iloc[curve-1:curve,:],smooth_to_lr(shape, curve, mob_start, mob_end).iloc[curve-1:curve,:], 'smooth to linear regression', 4)
    plot_curve_sbs_method(shape.iloc[curve-1:curve,:],log_plot(shape, curve, mob_start, mob_end).iloc[curve-1:curve,:], 'logarithm', 5)

    # if (mob_end - mob_start) >= 5:
    #     plot_curve_sbs_method(shape, poly_plot(shape, curve, mob_start, mob_end).iloc[curve-1:curve,:],'fourth degree polynomial', 6)

    # else:
    #     pass

def plot_curve_sbs_method(actual, df, title, k):
    ax1 = plt.subplot(2,4,k)
    for i in range(len(df)):
        plt.plot(actual.iloc[i], label = 'Actual')
        plt.plot(df.iloc[i], label = 'Smoothed', linestyle = 'dashed')
    ax1.set_title(title, color = 'black', fontsize = 10)
    ax1.legend(loc='lower right', fontsize = 6)

def smooth_to_average_mob(df, curve, mob_start, mob_end):
    df1 = df.copy()
    df1.iloc[curve-1:curve, mob_start-1:mob_end] = df.iloc[curve-1:curve, mob_start-1:mob_end].mean(axis = 1).values
    return df1


def smooth_to_ma_mob(df, curve, mob_start, mob_end, roll_window):
    df1 = df.copy()
    df1.iloc[curve-1:curve, mob_start:mob_end] = df.iloc[curve-1:curve, mob_start:mob_end].rolling(axis = 1, min_periods = 1, window = roll_window).mean().fillna(0)
    return df1

def smooth_to_max_mob(df, curve, mob_start, mob_end):
    df1 = df.copy()
    df1.iloc[curve-1:curve, mob_start-1:mob_end] = df.iloc[curve-1:curve, mob_start-1:mob_end].cummax(axis = 1)
    return df1

def slope(x1, y1, x2, y2):
    s = (y2-y1)/(x2-x1)
    return s
def intercept(x1, y1, x2, y2):
    s = slope(x1, y1, x2, y2)
    b = -x1*s+y1
    return b


def get_slope_intercept(df, row, mob_start, mob_end):
    para_dict = {}
    para_dict['slope'] = slope(mob_start, df.values[row][mob_start-1], mob_end, df.values[row][mob_end-1])
    para_dict['intercept'] = intercept(mob_start, df.values[row][mob_start-1], mob_end, df.values[row][mob_end-1])
    return para_dict
def smooth_to_lr(df, curve, mob_start, mob_end):
    df1 = df.copy()
    df1.values[curve-1][mob_start-1:mob_end] = df1.columns[mob_start-1:mob_end]*get_slope_intercept(df1, curve-1, mob_end,mob_start)['slope'] + get_slope_intercept(df1, curve-1, mob_end,mob_start)['intercept']
    return df1

def log_plot(df, curve, mob_start, mob_end):
    df_new = df.copy()
    pre_index = df.iloc[curve-1:curve, mob_start-1:mob_end].T
    seg = df.iloc[curve-1:curve, mob_start-1:mob_end].index.values[0]
    y = pre_index[seg]
    x = np.array(range(mob_start, mob_end+1))

    logy = np.polyfit(np.log(x), y, 1)
    new_y = []

    for j in x:
        new_y.append(logy[1] + (logy[0])*(np.log(j)))

    df_new.iloc[curve-1:curve, mob_start-1:mob_end] = new_y
    return df_new


def relevel_by_shape(sample_size, threshold, actual, shape, shape_mob_start, shape_mob_end, actual_mob_start, actual_mob_end):
    actual_c = actual.copy()
    for i in sample_size.index:
        if sample_size.loc[i].values <= threshold:
            factor = actual.loc[i,actual_mob_start:actual_mob_end].mean() / merge_curve_shape(actual, shape).loc[i, shape_mob_start, shape_mob_end].mean()
            actual_c.loc[i] = merge_curve_shape(actual, shape).loc[i,:actual_mob_end] * factor
    return actual_c


def merge_curve_shape(actual, shape):
    df1 = pd.DataFrame(index = actual.index)
    merged = pd.merge(df1, shape, left_index=True, right_index=True).reorder_levels(order = actual.index.names)
    return merged


def exp_function3(x,a,b,c):
    return a * np.exp(b*x) + c

def exp_function2(x,a,b):
    return a * np.exp(b*x)

def exp_function1(x,a):
    return a * np.exp(x)

def compute_distance(arr):
    diff = np.diff(arr)
    distance = np.insert(diff,0,0)

    v = distance[~np.isnan(distance)][-1]
    distance[np.isnan(distance)] = v
    return distance

def linear_function2(x,a,b):
    return a * x+ b

def linear_function1(x,a):
    return a * x


def fill_next_entry(df):
    for col in df.columns:
        last_value = None
        for i in df.index.values:
            if pd.notnull(df.loc[i,col]):
                last_value = df.loc[i,col]
            elif last_value is not None:
                df.loc[i,col] = last_value
    return df 


def extend(df, start, end):
    df_extend = pd.DataFrame(index = df.index, columns = range(start, end))
    df_merge = df.join(df_extend)
    df_merge = df_merge.fillna(0)
    return df_merge


def m_average(df, forecast, end):
    y = [i for i in range(df.columns[0], forecast+1)]
    df_ma = df.loc[:,y].copy()
    df_ma_ext = extend(df_ma, forecast+1, end+1)
    for i in df_ma_ext.columns:
        if i > forecast:
            df_ma_ext[i] = np.sum(df_ma_ext.loc[:, [i-1, i-2, i-3]], axis = 1) / 3
        return df_ma_ext

def m_average_ft(df, forecast, end, factor):
    y = [i for i in range(df.columns[0], forecast+1)]
    df_ma = df.loc[:,y].copy()
    df_ma_ext = extend(df_ma, forecast+1, end+1)
    for i in df_ma_ext.columns:
        if i > forecast:
            df_ma_ext[i] = np.sum(df_ma_ext.loc[:, [i-1, i-2, i-3]], axis = 1) * factor / 3
        return df_ma_ext
    

def forecast_lastmth(df, forecast, end):
    for i in range(forecast+1, end+1):
        df[i] = df[forecast]
    return df

def forecast_annlfee_ft(df, forecast, end, factor):
    y = [i for i in range(df.columns[0], forecast+1)]
    df_ma = df.loc[:,y].copy()
    df_ma_ext = extend(df_ma, forecast+1, end+1)
    for i in df_ma_ext.columns:
        if i > forecast:
            df_ma_ext[i] = np.sum(df_ma_ext.loc[:,[i-12]], axis = 1)* factor
    return df_ma_ext




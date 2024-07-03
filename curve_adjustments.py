from utils import *
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit



def view_grid(df):
    df = df.pivot_table(index = [s for s in seg if s != rev_seg], columns = rev_seg, dropna = False)
    df = df.reindex(rev_order, level = rev_seg, axis = 1)
    df = df.reindex(risk_order, level = risk_seg)
    return df.dropna(how = 'all')

def plot_grid(data, enclude_closed = True):

    if enclude_closed:
        data = data[data['flag'] != -1]['Target']
    else:
        data = data['Target']

    major_seg = [s for s in seg if s not in [risk_seg, rev_seg]]
    iterables = [data.reset_index()[col].dropna().to_list() for col in major_seg]
    major_index = pd.MultiIndex.from_product(iterables, names = major_seg).unique()

    for s in major_index:
        plt.figure(figsize=(6,4))
        plt.plot(data.loc[s])
        plt.legend(data.loc[s].columns)
        plt.title(s)
        for c in data.loc[s].columns:
            for x, y in zip(data.loc[s].index, data.loc[s][c]):
                text = f'{y:.4f}'
                plt.annotate(text, (x,y), textcoords='offset points', xytext=(0,10), ha = 'center')


##### power analysis###
threshold = 200

var = ['booked_accts',
       'open_accts',
       'mthly_writeoffs',
       'balance',
       'spend',
       'revolve_balance',
       'credit_limit',
       'annual_fee']

default_table = {'lmt_per_open': [1,16,1000],
                 'ulr_per_open': [9,16,0.0001],
                 'bal_util': [9,16,0.01],
                 'pur_util': [9,16,0.01],
                 'revol_util': [9,16,0.01],
                 'open_per_booked': [1,16,0.95],
                 'annual_fee': [1,16,5]}


class GetData:
    def __init__(self):
        self.raw_perf_12mob, self.raw_perf_shape, self.seg_vars = self.load_and_clean ()
        self.var_dict = self.calc_var ()
        self.sample_size_actual = self.sample_size_actual()

        self.per_open_metric_raw = self.per_open_metric_raw(self.var_dict)
        self.per_open_metric_raw_full = self.fill_with_defualts(default_table, threshold)

    def load_and_clean(self):
        raw_perf_12mob= pd.read_csv(input_path + Performance_12mob)
        raw_perf_12mob = cleaning(raw_perf_12mob)
        raw_perf_shape = pd.read_csv(input_path + Performance_shape)
        raw_perf_shape = cleaning(raw_perf_shape)
        seg_vars = rankings.drop (['risk_ranking','rev_ranking'], axis = 1)
        return raw_perf_12mob, raw_perf_shape, seg_vars
    
    def calc_var(self):
        var_dict = {}
        for x in var:
            var_dict[x] = pivot_out_full(self.raw_perf_12mob, x, seg)
        return var_dict
                

    def sample_size_actual(self):
        sample_size_actual = metrics_cal(aggregate(self.raw_perf_12mob, var, seg))['open_accts']
        sample_size_actual = pd.merge(self.seg_vars, sample_size_actual, how= 'left', left_index= True, right_index= True).drop('if_closed', axis = 1).fillna(0)
        return sample_size_actual
    
    def per_open_metric_raw(self, var_dict):
        per_open_metric_raw = {}
        per_open_metric_raw['ulr_per_open'] = var_dict['mthly_writeoffs']/var_dict['open_accts']
        per_open_metric_raw['bal_util'] = var_dict['balance']/var_dict['credit_limit']
        per_open_metric_raw['pur_util'] = var_dict['spend']/var_dict['credit_limit']
        per_open_metric_raw['revol_util'] = var_dict['revolve_balance']/var_dict['balance']
        
        per_open_metric_raw['lmt_per_open'] = var_dict['credit_limit']/var_dict['open_accts']
        per_open_metric_raw['open_per_booked'] = var_dict['open_accts']/var_dict['booked_accts']
        per_open_metric_raw['annual_fee'] = var_dict['annual_fee']/var_dict['open_accts']
        return per_open_metric_raw

    def fill_with_defualts(self, default, threshold) :
        metric_dict = self.per_open_metric_raw.copy()
        sample_size_actual = self.sample_size_actual

        for metric in metric_dict:
            metric_dict[metric] = pd.merge(self.seg_vars, metric_dict[metric], how = 'left', left_index = True, right_index = True)
        data = metric_dict.copy()
        for key in default:
            no_sample_index = sample_size_actual[sample_size_actual['valuesum'] == 0].index
            nan_index = data[key][data[key].drop('if_closed', axis = 1).sum (1) == 0].index
            index = no_sample_index & nan_index
            if len(index) != 0: 
                data[key].loc[index,default[key][0]:default[key][1]] = default[key][2]
                data[key].loc[nan_index, default[key][0]:default[key][1]] = default[key][2]
            if key == 'ulr_per_open':
                low_sample_index = sample_size_actual[sample_size_actual['valuesum'] < threshold].index
                no_ulr_index = data[key][data[key].drop('if_closed', axis = 1).sum (1) == 0].index
                index = low_sample_index & no_ulr_index
                if len(index) != 0: 
                    data[key].loc[index, default[key][0]:default[key][1]] = default[key][2]
        return data


    
# power analysis (TBD)
 
# leverage vintage with longer window to detect trend in the tail
class ShapeSelection():
    def __init__(self):
        self.shapes = {}
    def seg_combinations(self):
        combinations = []
        for r in range(len(seg)+1):
            for combination in itertools.combinations(seg, r):
                combinations.append(combination)
        combinations.pop(0)

        for y in range(len(combinations)):
            combinations[y] = list(combinations[y])
            combinations[y].append('MOB')
        return combinations
    def rollup(self):
        if self.mob == 16:
            self.perf = GetData().raw_perf_12mob
        else: 
            self.perf = GetData().raw_perf_shape

        combinations = self.seg_combinations()
        roll_up_set = [item for item in combinations if len(item) == self.rollup_level+1]
        rollup_metric_set = {}
        for i, segs in enumerate(roll_up_set):
            rollup_metric = metrics_cal(df_preprocess(self.perf, var, segs))[self.metric_name]
            rollup_metric_set[i] = rollup_metric
        return roll_up_set, rollup_metric_set
    def plot_shape(self, metric_name, rollup_level, mob):
        self.metric_name = metric_name
        self.rollup_level = rollup_level
        self.mob = mob

        roll_up_set, rollup_metric_set = self.rollup()
        mob_value = list(self.perf['MOB'].unique())

        if math.ceil(len(roll_up_set)/len(seg))>1:
            fig_size = (20,10)
        else:
            fig_size = (20,4)
        fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size,dpi = 180)

        for i, segs in enumerate(roll_up_set):
            ax1 = plt.subplot(math.ceil(len(roll_up_set)/len(seg)), len(seg), i+1)
            rollup_metric = rollup_metric_set[i]
            for j in range(len(rollup_metric)):
                plt.plot(mob_value, rollup_metric.iloc[j], label = rollup_metric.index.values[j])
                ax1.set_title(self.metric_name,color='black', fontsize=10)
                ax1.legend(loc='upper right', fontsize=6)
    def select_shape(self, n):
        rollup_metric_set = self.rollup()[1]

        shape = rollup_metric_set[n-1]

        self.shapes[self.metric_name] = shape

        return shape

    def save_shape(self):
        return self.shapes





# add advanced smoothing methodology
class Smoothing:
    def __init__(self, cls_instance):
        self.cls_instance = cls_instance
        self.all_smoothened_shapes = {}
    def get_shape(self):
        shape = self.cls_instance.shapes[self.metric_name]
        return shape
    def view_all_methods(self, metric_name, curve, mob_start, mob_end):
        self.metric_name = metric_name 
        self.shape = self.get_shape()
        self.smoothened_shape = self.shape.copy()
        view_all_smooth_method(self.shape, curve, mob_start, mob_end)

    def select_method(self, smooth_method, curve, mob_start, mob_end):
        if smooth_method == 'smooth to average level':
            smoothened_shape = smooth_to_average_mob(self.smoothened_shape, curve, mob_start, mob_end)
        elif smooth_method == 'smooth to moving average':
            rolling_window = 3
            smoothened_shape = smooth_to_ma_mob(self.smoothened_shape, curve, mob_start, mob_end, rolling_window)
        elif smooth_method == 'smooth to max':
            smoothened_shape = smooth_to_max_mob(self.smoothened_shape, curve, mob_start, mob_end)
        elif smooth_method == 'smooth to linear regression':
            smoothened_shape = smooth_to_lr(self.smoothened_shape, curve, mob_start, mob_end)
        elif smooth_method == 'logarithm':
            smoothened_shape = log_plot(self.smoothened_shape, curve, mob_start, mob_end)
        else:
            print('wrong method name, please retry.')

        self.smoothened_shape = smoothened_shape
        self.all_smoothened_shapes[self.metric_name] = self.smoothened_shape
        return self.smoothened_shape
    def save_smoothened_shape(self):
        return self.all_smoothened_shapes



class Releveling:
    def __init__(self, cls_instance):
        self.cls_instance = cls_instance
        self.sample_size_actual = GetData().sample_size_actual
        self.all_releveled_metrics = {}

    def get_smoothened_shape(self):
        smoothened_shape = self.cls_instance.all_smoothened_shapes[self.metric_name]
        return smoothened_shape

    def relevel(self, metric_name, threshold, shape_mob_start, shape_mob_end, actual_mob_start, actual_mob_end):
        self.metric_name = metric_name
        smoothened_shape = self.get_smoothened_shape()
        metric = GetData().per_open_metric_raw_full[self.metric_name].drop('if_closed', axis = 1)

        self.releveled_metric = relevel_by_shape(self.sample_size_actual, threshold, metric, smoothened_shape, shape_mob_start, shape_mob_end, actual_mob_start, actual_mob_end)
        self.all_releveled_metrics[self.metric_name] = self.releveled_metric
        return self.releveled_metric
    def save_releveled_metric(self):
        return self.all_releveled_metrics


# borrow the idea of machine leanring, learn rank ordered levels and predict/adjust unrank ordered levels
class RankOrdering:
    def __init__(self, cls_instance):
        self.cls_instance = cls_instance
        self.sample_size_actual = GetData().sample_size_actual
        self.rankings = rankings
        self.rank_order_levels = {}

    def get_relevel(self):
        all_releveled_metrics = self.cls_instance.all_releveled_metrics
        return all_releveled_metrics
    
    def calc_avg(self):
        metric = self.get_relevel()[self.metric_name]
        metric_avg = metric.loc[:, self.start_MOB:self.end_MOB].mean(axis = 1).reset_index().rename(columns ={0: 'Target'}).set_index(seg)
        return metric_avg
    
    def flagging(self, df):
        rev_indices = [i for i in range(len(seg)) if i not in [seg.index(rev_seg)]]
        risk_indices = [i for i in range(len(seg)) if i not in [seg.index(risk_seg)]]
        df['actual_risk_ranking'] = df[df['if_closed']==0].groupby(level = risk_indices)['Target'].rank(method='first', ascending = True)
        df['actual_rev_ranking'] = df[df['if_closed']==0].groupby(level = rev_indices)['Target'].rank(method='first', ascending = True)
        
        df['flag'] = ~ ((df['actual_rev_ranking'] == df['rev_ranking']) & (df['actual_risk_ranking'] == df['risk_ranking']))
        df['flag'] = df['flag'] * 1
        df.loc[df['if_closed'] == 1, 'flag'] = -1
        return df
    
    def encoding(self):
        df = pd.concat([self.calc_avg(), self.sample_size_actual, self.rankings], axis = 1)
        df = self.flagging(df)
        encoded_data = pd.get_dummies(df.reset_index(), columns=seg, drop_first=True)
        encoded_data = pd.concat([df.reset_index()[seg], encoded_data], axis = 1).set_index(seg)
        return encoded_data
    
    def linear_regression(self, encoded_data):
        test = encoded_data[encoded_data['flag']==1]
        train = encoded_data[encoded_data['flag']!=1]

        sample_weight = train['valuesum']

        regressor = LinearRegression()
        seg_variables = [col for col in encoded_data.columns if col.startswith('SEGMENTATION_VAR_')]
        X = train[seg_variables]
        y = train['Target']

        regressor.fit(X, y, sample_weight=sample_weight)
        
        test['Target'] = regressor.predict(test[seg_variables])
        encoded_data.loc[test.index.values] = test
        encoded_data = self.flagging(encoded_data)
        return encoded_data
    
    def linear_regression_plus(self, metric_name, avg_start_MOB, avg_end_MOB, iteration):
        self.metric_name = metric_name
        self.start_MOB = avg_start_MOB
        self.end_MOB = avg_end_MOB
        self.n = iteration

        encoded_data = self.encoding()
        check = encoded_data[encoded_data['flag']!=-1]

        iterations = 1
        while (check['flag'].sum()!=0) & (iterations <= self.n):
            encoded_data = self.linear_regression(encoded_data)
            check = encoded_data[encoded_data['flag']!=-1]
            iterations += 1
        
        self.encoded_result = encoded_data[['Target', 'flag', 'valuesum']]
        self.rank_order_levels[self.metric_name] = self.encoded_result

        return view_grid(self.encoding()[['Target', 'flag', 'valuesum']]), view_grid(self.encoded_result)
    

    def save_rankorder_levels(self):
        return self.rank_order_levels
    
    def plot(self):
        result = view_grid(self.encoded_result)
        plot_grid(result)                             






# fullfill empty cells due to unqualitied credit
class RejectInferencing:
    def __init__(self, cls_instance):
        self.cls_instance = cls_instance
        self.reject_infer_levels = {}

    def get_rankorder_levels(self):
        rank_order_levels = self.cls_instance.rank_order_levels
        return rank_order_levels
    
    def reject_inferencing_ulr(self, reverse_rev = True):
        data = self.get_rankorder_levels()['ulr_per_open']
        data.rename(columns = {'flag': 'if_closed'}, inplace= True)
        data['if_closed'] = data['if_closed'].apply(lambda v:1 if v == -1 else 0)
        data.loc[data[data['if_closed']==1].index, 'Target'] = np.nan
        
        major_seg = [s for s in seg if s not in [risk_seg, rev_seg]]
        iterables = [data.reset_index()[col].dropna().to_list() for col in major_seg]
        major_index = pd.MultiIndex.from_product(iterables, names = major_seg).unique()

        table = pd.DataFrame()
        seg_view = view_grid(data)
        a = 0
        b = 0
        c = 0
        for s in major_index:
            grid = view_grid(data).loc[s]
            
            y_fit = grid['Target'].to_numpy()

            rows, cols = y_fit.shape
            x_risk, x_rev = np.indices((rows, cols)) + 1

            if reverse_rev:
                x_rev = np.flip(x_rev, axis = 1)
            
            n_risk = (rows - grid['if_closed'].sum(0)).astype(int)
            n_rev = (cols - grid['if_closed'].sum(1)).astype(int)

            y_risk = y_fit.copy()

            for i, n in enumerate(n_risk):
                n = int(n)

                if n > 2:
                    params, covariance = curve_fit(exp_function3, x_risk[:n, i], y_fit[:n,i], maxfev = 5000)
                    y_risk[n:, i] = exp_function3(x_risk[n:,i], *params)
                    a,b,c = params
                else: 
                    if i == 0 & n == 2:
                        params, covariance = curve_fit(exp_function2, x_risk[:n, i], y_fit[:n,i], maxfev = 5000)
                        y_risk[n:, i] = exp_function2(x_risk[n:,i], *params)
                        a,b = params
                        c = 0
                    elif i == 0 & n == 1:
                        params, covariance = curve_fit(exp_function1, x_risk[:n, i], y_fit[:n,i], maxfev = 5000)
                        y_risk[n:, i] = exp_function1(x_risk[n:,i], *params)
                        a = params
                        b = 1
                        c = 0
                    else:
                        if n_rev[0] < cols:
                            scale = compute_distance(y_fit[0])
                            c = c + scale[i]/[i+1]
                        else:
                            filled = y_risk[np.isnan(y_risk).sum(1)==0]
                            scale = compute_distance(filled.mean(0))
                            c = c + scale[i]*(i+1)
                        y_risk[n:,i] = exp_function3(x_risk[n:,i],a,b,c)

                        missing_indices = np.where(np.isnan(y_risk[:,i]))
                        y_risk[missing_indices, i] = exp_function3(x_risk[missing_indices,i],a,b,c)

                seg_view.loc[s,'Target'] = y_risk

            table = seg_view.stack().reset_index().set_index(seg)
            self.result = table
            self.reject_infer_levels['ulr_per_open'] = self.result
        return seg_view, table


    def reject_inferencing_rev(self, metric_name, reverse_rev = True):
        data = self.get_rankorder_levels()[metric_name]
        data.rename(columns = {'flag': 'if_closed'}, inplace= True)
        data['if_closed'] = data['if_closed'].apply(lambda v:1 if v == -1 else 0)
        data.loc[data[data['if_closed']==1].index, 'Target'] = np.nan
        
        major_seg = [s for s in seg if s not in [risk_seg, rev_seg]]
        iterables = [data.reset_index()[col].dropna().to_list() for col in major_seg]
        major_index = pd.MultiIndex.from_product(iterables, names = major_seg).unique()

        table = pd.DataFrame()
        seg_view = view_grid(data)
        a = 0
        c = 0
        for s in major_index:
            grid = view_grid(data).loc[s]
            
            y_fit = grid['Target'].to_numpy()

            rows, cols = y_fit.shape
            x_risk, x_rev = np.indices((rows, cols)) + 1

            if reverse_rev:
                x_rev = np.flip(x_rev, axis = 1)
            
            n_risk = (rows - grid['if_closed'].sum(0)).astype(int)
            n_rev = (cols - grid['if_closed'].sum(1)).astype(int)

            y_rev = y_fit.copy()

            for i, n in enumerate(n_rev):
                n = int(n)

                if n >= 2:
                    params, covariance = curve_fit(linear_function2, x_rev[i, :n], y_fit[i, :n])
                    y_rev[i, n:] = linear_function2(x_rev[i, n:], *params)
                    a,c = params
                else: 
                    if i == 0 & n == 1:
                        params, covariance = curve_fit(linear_function1, x_rev[i, :n], y_fit[i, :n])
                        y_rev[i, n:] = linear_function1(x_rev[i, n:], *params)
                        a = params
                        c = 0
                    else:
                        if n_risk[0] < rows:
                            scale = compute_distance(y_fit[:,0])
                            c = c + scale[i]
                        else:
                            filled = y_rev[np.isnan(y_rev).sum(0)==0]
                            scale = compute_distance(filled.mean(0))
                            c = c + scale[i]
                        y_rev[i,n:] = linear_function2(x_rev[i,n:],a,c)

                        missing_indices = np.where(np.isnan(y_rev[i]))
                        y_rev[i, missing_indices] = linear_function2(x_rev[i, missing_indices],a,c)

                y_rev = np.where(y_rev>0.95, 0.95, y_rev)
                seg_view.loc[s,'Target'] = y_rev

            table = seg_view.stack().reset_index().set_index(seg)
            self.result = table
            self.reject_infer_levels[metric_name] = self.result
        return seg_view, table

    def save_rejectinfer_levels(self):
        return self.reject_infer_levels
    
    def plot(self):
        grid_result = view_grid(self.result)
        plot_grid(grid_result, False)


# add visualization
class FinalMetrics:
    def __init__(self, cls_instance_relevel, cls_instance_rejectinfer):
        self.cls_instance_relevel = cls_instance_relevel
        self.cls_instance_rejectinfer = cls_instance_rejectinfer
        self.closed_index = rankings.loc[rankings['if_closed']==1].index.values
        self.final_adjusted_metrics = {}

    def get_relevel(self):
        all_releveled_metrics = self.cls_instance_relevel.all_releveled_metrics
        return all_releveled_metrics

    def get_rejectinfer_levels(self):
        reject_infer_levels = self.cls_instance_rejectinfer.reject_infer_levels
        return reject_infer_levels
    
    def get_cell_level_metrics(self):
        releveled_metrics = self.get_relevel()
        reject_infer_levels = self.get_rejectinfer_levels()
        for metric_name in releveled_metrics:
            if metric_name not in ['ulr_per_open', 'bal_util', 'pur_util', 'revol_util', 'lmt_per_open']:
                releveled_metrics[metric_name].loc[self.closed_index] = None
                releveled_metrics[metric_name] = fill_next_entry(releveled_metrics[metric_name])
            elif metric_name in ['ulr_per_open', 'bal_util', 'pur_util', 'revol_util']:
                metric = releveled_metrics[metric_name]
                avg = reject_infer_levels[metric_name]
                zero_index = metric[metric.loc[:,10:12].mean(axis = 1) == 0].index.values
                metric.loc[zero_index, 10:12] = 1
                factor = avg['Target']/metric.loc[:, 10:12].mean(axis = 1)
                metric = metric.mul(factor, axis = 0)
                releveled_metrics[metric_name] = metric
                
        self.final_adjusted_metrics = releveled_metrics
        return releveled_metrics
    
    def save_final_metrics(self):
        return self.final_adjusted_metrics
    

    





class Forecasting:
    def __init__(self, cls_instance):
        self.cls_instance = cls_instance
        #define forecasting methods for each metric
        self.forecast_method = {m_average_ft: ['lmt_per_open'],
                                forecast_lastmth: ['bal_util'],
                                forecast_annlfee_ft: ['annual_fee'],
                                m_average: ['pur_util', 'revol_util', 'ulr_per_open']}
        
        self.factor = {'lmt_per_open': 1.1,
                       'annual_fee': 0.9}
        self.output = {}

    def get_cell_level_metrics(self):
        final_adjusted_metrics = self.cls_instance.final_adjusted_metrics
        return final_adjusted_metrics

    def forecast(self):
        metric_60mob = {}
        open_per_booked_mom = {}

        per_open_metric_60mob = self.get_cell_level_metrics()
        open_metrics = [metric for metric in per_open_metric_60mob.keys() if 'booked' not in metric]
        booked_metrics = [metric for metric in per_open_metric_60mob.keys() if 'booked' in metric]
        for metric in open_metrics:
            forecast_func = [key for key, value in self.forecast_method.items() if metric in value][0]
            if metric == 'lmt_per_open' or metric == 'annual_fee':
                metric_60mob[f'{metric}_60mob'] = forecast_func(per_open_metric_60mob[metric], 12, 60, self.factor[metric])
            else:
                metric_60mob[f'{metric}_60mob'] = forecast_func(per_open_metric_60mob[metric], 12, 60)

        for metric in booked_metrics:
            open_per_booked_mom[metric]= self.mom_factor(booked_metrics[metric])
            metric_60mob[f'{metric}_60mob'] = self.num_open_accounts(booked_metrics[metric], open_per_booked_mom[metric])

        return metric_60mob

    def mom_factor(self, df):
        # month over month factor
        df_mom = df.copy()
        for i in df.index:
            for j in df_mom.loc[i].index:
                if j == 1:
                    df_mom.loc[i][j] = 0
                if j > 1:
                    df_mom.loc[i][j] = df.loc[i][j]/df.loc[i][j-1]
        li = []
        for i in df_mom.index:
            li.append(sum(df_mom.loc[i][6:12])/6)
        df_mom['factor']=li
        return df_mom
    
    def num_open_accounts(self, restate_cnt, open_per_booked_mom):
        num_open_accounts_60mob = extend(restate_cnt,13,61).astype('float64')
        for i in num_open_accounts_60mob.index:
            for j in num_open_accounts_60mob.loc[i].index:
                if j > 12:
                    num_open_accounts_60mob.loc[i][j] = num_open_accounts_60mob[i][j-1]*open_per_booked_mom.loc[i]['factor']
        return num_open_accounts_60mob
    
    def save_forecast_output(self):
        self.output = self.forecast()
        return self.output
    
    def export_to_excel(self):
        with pd.ExcelWriter(output_path + 'forecast_60mob.xlsx') as writer:
            for key, val in self.output.items():
                val.to_excel(writer, sheet_name = key)



class Validate(FinalMetrics):
    def __init__(self):
        self.shapes = ShapeSelection().shapes
        self.historicals = pd.read_excel()
        self.roll_up = self.roll_up()

    def roll_up(self):
    #leverage sample size anf weighted average cell level metrcis and compare with shapes
        return
    def reweighting(self):
    # shift historicals distribution to align with most updated samples
        return
    def model_risk(self):
        unfavourable_factor = 1 
        favourable_factor = 0.5
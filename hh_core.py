import pandas as pd
import numpy as np
import math
import gc

### MultiIndex level slice constant:
every = slice(None)

### NA for MS Excel files:
list_na_excel_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null', '#N/A Requesting Data...', '#N/A Invalid Security', '#N/A Field Not Applicable']

def gen_source_dataset(str_date_start='2000-01-01', str_date_end=(pd.Timestamp.now() - pd.DateOffset(years=1)).replace(month=12, day=31).strftime('%Y-%m-%d'), str_freq='B', list_cols=None, dict_levels=None, dict_nans=None, dict_distributions=None):
    '''
    Generate a DataFrame with random numerical data indexed by a date range and additional index levels.
    
    This function creates a pandas DataFrame filled with random values.
    The index is a multi-index that includes a date range from `str_date_start` to `str_date_end` with a given `str_freq`,
    along with additional levels specified in `dict_levels`. NaN values can be added per column with `dict_nans`.
    
    Parameters:
        str_date_start (str): The start date in YYYY-MM-DD format.
        str_date_end (str): The end date in YYYY-MM-DD format.
        str_freq (str): The frequency of the date range (e.g., 'D' for daily, 'M' for monthly). Default is 'B' (business days).
        list_cols (list of str): The list of column names. Default is ['Some_Data'].
        dict_levels (dict): Dictionary where keys are additional index level names and values are lists of possible values.
        dict_nans (dict): Dictionary where keys are column names and values define the number of NaNs:
                         - If integer, the exact number of NaNs to insert.
                         - If float (0-1), the percentage of NaNs to insert.
        dict_distributions (dict): Dictionary where keys are column names and values define the random distribution:
                         - 'normal' with (mean, std_dev) tuple.
                         - 'uniform' with (low, high) tuple.
                         Default is normal(0, 0.1) for all columns.
        
    Returns:
        pd.DataFrame: A DataFrame with shape (num_rows * num_combinations, len(list_cols)), indexed by dates and additional levels.
    '''
  
    try:
        from itertools import product
        # Default column names
        if list_cols is None:
            list_cols = ['Some_Data']
        
        # Default distributions
        if dict_distributions is None:
            dict_distributions = {col: ('normal', (0, 0.1)) for col in list_cols}
        
        # Generate date index
        idx_dates = pd.date_range(start=str_date_start, end=str_date_end, freq=str_freq)
        num_rows = len(idx_dates)
        
        # Generate additional index levels
        if dict_levels:          
            list_level_names = list(dict_levels.keys())
            idx_multi = pd.MultiIndex.from_product([idx_dates] + list(dict_levels.values()), names=['Date'] + list_level_names)
        else:
            idx_multi = pd.Index(idx_dates, name='Date')
        
        # Generate random data
        data_matrix = np.zeros((len(idx_multi), len(list_cols)))
        for i, col in enumerate(list_cols):
            dist_type, params = dict_distributions.get(col, ('normal', (0, 0.1)))
            if dist_type == 'normal':
                data_matrix[:, i] = np.random.normal(*params, len(idx_multi))
            elif dist_type == 'uniform':
                data_matrix[:, i] = np.random.uniform(*params, len(idx_multi))
        
        df_generated = pd.DataFrame(data_matrix, columns=list_cols, index=idx_multi)
        
        # Apply NaN values if dict_nans is provided
        if dict_nans:
            for col, nan_value in dict_nans.items():
                if col in df_generated.columns:
                    if isinstance(nan_value, int):
                        nan_indices = np.random.choice(df_generated.index, size=min(nan_value, len(df_generated)), replace=False)
                    elif isinstance(nan_value, float) and 0 <= nan_value <= 1:
                        num_nans = int(nan_value * len(df_generated))
                        nan_indices = np.random.choice(df_generated.index, size=num_nans, replace=False)
                    else:
                        continue
                    df_generated.loc[nan_indices, col] = np.nan
        return df_generated

    except Exception as err:
        print('Error generating DataFrame: %s', str(err))
        return None

### DEFINING COUNTRY CODES EXTRACTOR
def get_country_codes(use_local_copy = False):  
    ### In case if URL is unavailable:
    if (use_local_copy):
        url_country_code = 'Data_Files/Source_Files/countrycode.html'
    ### Online extraction:
    else:
        url_country_code = 'https://countrycode.org/'
    df_full_codes = pd.read_html(url_country_code, index_col = 'COUNTRY')[0]
    df_full_codes[['ISO SHORT', 'ISO LONG']] = df_full_codes['ISO CODES'].str.split(' / ', expand = True)
    df_result = df_full_codes[['ISO SHORT', 'ISO LONG']].sort_index()    
    df_result.index = df_result.index.str.upper()
    ### Results output:
    return df_result

### DEFINING EXTRACTION UNIVERSE DATA FROM MS EXCEL SOURCE
def ison_membership_converting(str_path_universe, date_end, bool_daily = False, int_backfill_months = 0):
    ### Defining business-month-end reindexation on country level:
    def country_modify(ser_raw_country, date_end):
        ser_res_country = ser_raw_country.droplevel(0).resample('MS').last().resample('BM').last()
        range_country = pd.date_range(ser_res_country.index[0], date_end, freq = 'BM')
        return ser_res_country.reindex(range_country).ffill()
    ### Markets encoding table:
    dict_markets = {50 : 'DM', 57 : 'EM', 504 : 'FM', 0: np.NaN}     
    ### Loading source file:
    df_raw_universe = pd.read_excel(engine = 'openpyxl', io = str_path_universe, sheet_name = 'Switchers', header = 0, index_col = [0, 1],
                                    parse_dates = [1, 2], date_format='%m/%d/%Y', 
                                    na_values = list_na_excel_values, keep_default_na = False)
    ### Converting source file:
    df_raw_universe.index.names = ['Country', 'Date']
    ser_raw_universe = df_raw_universe['Region']
    ser_raw_universe.fillna(0, inplace = True)
    ser_raw_universe.name = 'Market'
    ### By country reindexation and translation:
    ser_res_universe = ser_raw_universe.groupby('Country').apply(country_modify, date_end)
    ser_res_universe.index.names = ['Country', 'Date']
    ser_res_universe = ser_res_universe.replace(dict_markets).reorder_levels([1, 0]).sort_index() 
    ### Expanding membership for primary regions members by backfilling:
    if int_backfill_months:
        ### List of regions:
        list_region = list(ser_res_universe.dropna().unique())
        ### Initialising of collection of series with backfilled data for each region:
        list_ison_backfill = []
        ### Regions looping:
        for iter_region in list_region:
            ### Defining start of region date:
            date_first_valid = ser_res_universe.loc[ser_res_universe == iter_region].first_valid_index()[0]
            ### Creating dates index to backfilling:
            idx_date_backfill = pd.date_range(end = date_first_valid, periods = int_backfill_months + 1, freq = 'BM')[: -1]
            ### Creating primary countries index to backfilling:            
            idx_region_backfill = ser_res_universe.loc[ser_res_universe == iter_region].loc[date_first_valid, All].index.get_level_values('Country')
            ### Creating full index:
            idx_ison_backfill = pd.MultiIndex.from_product([idx_date_backfill, idx_region_backfill])
            ### Series with backfilled data:
            list_ison_backfill.append(pd.Series(iter_region, index = idx_ison_backfill))
        ### Combination of backfilled series and original ISON data:    
        ser_res_universe = ser_res_universe.combine_first(pd.concat(list_ison_backfill, axis = 0)).sort_index()  
        ser_res_universe.index.names = ['Date', 'Country']
    ### Converting to daily frequency:
    if bool_daily:
        ser_res_universe = ser_res_universe.reset_index('Country').groupby('Country').resample('B').ffill()['Market'].swaplevel().sort_index()    
    ### Results output:
    ser_res_universe.name = 'Market'
    return ser_res_universe    

### DEFINING WEIGHTED AVERAGE FOR DATAFRAME COLUMNS
def columns_average(df_series, list_weights = False): 
    ### Single column check
    if (len(df_series.columns) > 1):
        ### Equal weights list creating:
        if isinstance(list_weights, bool):
            list_weights = [1] * len(df_series.columns)
        ### Dataframe of weights initialising:
        df_weights = pd.DataFrame(np.NaN, index = df_series.index, columns = df_series.columns)
        for iter_num, iter_col in enumerate(df_weights.columns):
            df_weights[iter_col] = list_weights[iter_num]
        ### Zeroing weights for NaN values:
        for iter_col in df_weights.columns:
            df_weights.loc[df_series[iter_col].isna(), iter_col] = 0
        ser_mean = (df_series.multiply(df_weights).sum(axis = 1)).div(df_weights.sum(axis = 1))    
        ### Results output:
        del df_series
        del df_weights    
        gc.collect()
    else:
        ser_mean = df_series.squeeze()
        del df_series
        gc.collect()        
    return ser_mean

### DEFINING ACADIAN STYLE TWO-STEP FACTOR VECTOR STANDARTIZATION
def td_two_stage_standardize(ser_factor):
    ### Limits definition:
    flo_trunc_limit_1 = 2.5
    flo_trunc_limit_2 = 2.0
    ### Preliminary statistics calculation:
    flo_std = np.nanstd(ser_factor, axis = 0, ddof = 1)
    flo_mean = np.nanmean(ser_factor, axis = 0)
    ### Preliminary z-scoring:
    ser_score = (ser_factor - flo_mean) / flo_std
    ### Constant vector checking:
    if np.isclose(flo_std, 0.0):
        ser_score = ser_factor - ser_factor
    ### First winsorization step:
    ser_score.loc[ser_score < (-1.0 * flo_trunc_limit_1)] = -1.0 * flo_trunc_limit_1
    ser_score.loc[ser_score > flo_trunc_limit_1] = flo_trunc_limit_1
    ### First limit precence marker:
    ser_on_limit = (ser_score.abs() == flo_trunc_limit_1)
    ### Check if first step do some truncation:    
    if ser_on_limit.any():
        ### Under the limit values marking:
        ser_off_limit = (ser_score.abs() != flo_trunc_limit_1)
        ### Separating truncated values to perform further transformations with under the limit values only:
        ser_score_trunc_1 = ser_score.copy()
        ser_score_trunc_1.loc[ser_off_limit] = 0.0
        ### Dropping truncaterd values for further performance:
        ser_score.loc[ser_on_limit] = np.NaN
        ### Repeated statistics calculation:
        flo_std = np.nanstd(ser_score, axis = 0, ddof = 1)
        flo_mean = np.nanmean(ser_score, axis = 0)
        ### Constant vector checking:
        if np.isclose(flo_std, 0.0):
            ser_score = ser_score - ser_score
        else:
            ### Second z-scoring:
            ser_score = (ser_score - flo_mean) / flo_std
        ### Dropping truncaterd values for further performance:
        ser_score.loc[ser_on_limit] = np.NaN
        ### Second winsorization step:  
        ser_score.loc[ser_score < (-1.0 * flo_trunc_limit_2)] = -1.0 * flo_trunc_limit_2
        ser_score.loc[ser_score > flo_trunc_limit_2] = flo_trunc_limit_2
        ### Preparing for truncated values adding:
        ser_score.loc[ser_on_limit] = 0.0
        ### Vectors union:
        ser_score = ser_score + ser_score_trunc_1
        ### Final demean:
        flo_mean = np.nanmean(ser_score, axis = 0)  
        ser_score = ser_score - flo_mean
    ### Results output:
    return ser_score

### DEFINING EXPONENTIAL WEIGHT
def exp_weight_single(halflife_len = 3, num_element = 0):
    ### Weight calculating:
    num_period_factor = math.exp(math.log(0.5) / round(halflife_len))
    num_weight = math.exp(math.log(num_period_factor) * num_element)
    ### Result output:
    return num_weight

### DEFINING WEIGHTED AVERAGE
def weighted_average(ser_data, ser_weight = None, int_min_count = 0, bool_normalize = True):
    ### Default output:
    num_result = np.NaN
    ### Checking for data presence:
    if (ser_data.count() > int_min_count):       
        ### Checking for weights dataset:
        if (ser_weight is None):
            ### Calculating of simple average:
            num_result = np.nanmean(ser_data.values)
        else:
            ### Weights filtering:
            list_weight = ser_weight[ser_data.dropna().index].values
            ### Checking for weights presence:
            if np.nansum(list_weight):
                ### Data filtering:
                list_data = ser_data.dropna().values
                ### Weighted average calculating:
                if bool_normalize:
                    num_result = np.nansum(list_data * list_weight) / np.nansum(list_weight)
                else:
                    num_result = np.nansum(list_data * list_weight)
    ### Results output:
    return num_result

### DEFINING AVERAGE OF RETURNS CALCULATION (TO IMPLEMENT ADDITIONAL CONTROL FILTERS)
def weighted_average_grouped(ser_country, ser_weight = None):
    ### Minimum number of observations:
    int_min_number = int(260 / 2)
    ### Last values control interval length:
    int_last_control = 10
    ### Implementing control filters and performing weighted average calulation:
    if ((ser_country.count() >= int_min_number) & (ser_country[-int_last_control : ].count() > 0)):
        flo_wa = weighted_average(ser_country.droplevel('Country'), ser_weight)
    else:
        flo_wa = np.NaN
    return flo_wa

### DEFINING MULTI-STEP STANDARTIZATION BY MARKET FOR CROSS-SECTION
def ison_standartize(ser_to_manage, arr_truncate, ser_weight = False, reuse_outliers = False, center_result = True, full_result = False, within_market = False):
    ### Multi-step standartizing:
    if (within_market):
    ### Within market standartizing:
        ser_result = ser_to_manage.groupby(by = 'Market', group_keys = False).apply(multistep_standartize, arr_truncate, ser_weight, 
                                                                                                  reuse_outliers, center_result, full_result)
    else:
    ### Full universe standartizing:
        ser_result = multistep_standartize(ser_to_manage, arr_truncate, ser_weight, reuse_outliers, center_result, full_result)
    ### Results output:
    return ser_result

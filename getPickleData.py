import pandas as pd
import numpy as np
from itertools import chain
from scipy import interpolate

# Script to load the wesad data from the pickle files and upsample them to 700 hz
# Note: check file paths before running and potentially overwriting old files!

# This function basically un-nests the nested arrays within the pickle format
def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def get_pickle_data(path):
    data = pd.read_pickle(path)

    df = pd.DataFrame()

    # Labels
    df['Label'] = data['label']

    # Chest data
    df['ACC_chest_x'] = data['signal']['chest']['ACC'][:,0]
    df['ACC_chest_y'] = data['signal']['chest']['ACC'][:,1]
    df['ACC_chest_z'] = data['signal']['chest']['ACC'][:,2]
    df['EMG_chest'] = list(flatten(data['signal']['chest']['EMG']))
    df['ECG_chest'] = list(flatten(data['signal']['chest']['ECG']))
    df['EDA_chest'] = list(flatten(data['signal']['chest']['EDA']))
    df['TEMP_chest'] = list(flatten(data['signal']['chest']['Temp']))
    df['RESP_chest'] = list(flatten(data['signal']['chest']['Resp']))

    # Wrist data
    ACC_wrist_x = data['signal']['wrist']['ACC'][:,0]
    ACC_wrist_y = data['signal']['wrist']['ACC'][:,1]
    ACC_wrist_z = data['signal']['wrist']['ACC'][:,2]
    BVP_wrist = list(flatten(data['signal']['wrist']['BVP']))
    EDA_wrist = list(flatten(data['signal']['wrist']['EDA']))
    TEMP_wrist = list(flatten(data['signal']['wrist']['TEMP']))
    wrist_data = {'ACC_wrist_x': ACC_wrist_x, 'ACC_wrist_y': ACC_wrist_y, 
                  'ACC_wrist_z': ACC_wrist_z, 'BVP_wrist': BVP_wrist, 
                  'EDA_wrist': EDA_wrist, 'TEMP_wrist': TEMP_wrist}

    # Upsample
    resampled_wrist_data = upsample_data(wrist_data, df.shape[0])
    
    # There's a bit of data mismatch since 700 / 32 is not an integer
    # Here, we figure out what the most rows we can interpolate is
    min_row = np.inf
    for resampled_data in resampled_wrist_data:
        if len(resampled_wrist_data[resampled_data]) < min_row:
            min_row = len(resampled_wrist_data[resampled_data])
            
    # Here, we check if there's any data that we're throwing out
    # Probably won't throw any useful data out because it's near the end of the study
    if 2 in df.iloc[min_row:]['Label'].unique() or 3 in df.iloc[min_row:]['Label'].unique():
        print("WARNING: throwing out data! No rows added for path", path)
        df = pd.DataFrame()
    else: # if not, get rid of the extra data and add to df
        df = df.iloc[0:min_row]
        for resampled_data in resampled_wrist_data:
            resampled_wrist_data[resampled_data] = resampled_wrist_data[resampled_data][0:min_row]
            df[resampled_data] = resampled_wrist_data[resampled_data]

    return df


def upsample_data(wrist_data, num_samps):
    """
    Upsample the wrist data linearly and cubicly

    Args:
        wrist_data: dictionary that maps the name of the wrist data to the data
        num_samps: the final length of the resampled data
        
    Returns:
        dictionary that maps the same names to the resampled data
    """
    final_hz = np.linspace(0, 1, num_samps, endpoint=False)
    resampled_wrist_data = {}
    for wdat in wrist_data:
        current_hz = np.linspace(0,1, len(wrist_data[wdat]), endpoint=False)
        if wdat == 'TEMP_wrist':
            resampler = interpolate.interp1d(current_hz, wrist_data[wdat], kind='linear')
        else:
            resampler = interpolate.interp1d(current_hz, wrist_data[wdat], kind='cubic')
        resampled_wrist_data[wdat] = resampler(final_hz[final_hz <= current_hz[-1]])
    return resampled_wrist_data

def calc_slope(x):
#     slope = np.polyfit(range(len(x)), x, 1)[0]
    slope = (x[-1] - x[0]) / len(x)
    return slope

def calculate_window_stats(df, window_size):
    df_min = df.rolling(window_size).min()
    df_max = df.rolling(window_size).max()
    df_mean = df.rolling(window_size).mean()
    df_sd = df.rolling(window_size).std()
    df_range = df_max - df_min
    df_slope = df.rolling(window_size).apply(calc_slope, raw=True)

    # Rename columns
    df_mean.columns = df_mean.columns + '_mean'
    df_sd.columns = df_sd.columns + '_sd'
    df_range.columns = df_range.columns + '_range'
    df_slope.columns = df_slope.columns + '_slope'

    # Get necessary columns
    df_slope = df_slope[['EDA_chest_slope', 'TEMP_chest_slope', 'EDA_wrist_slope', 'TEMP_wrist_slope']]
    df_range = df_range[['EDA_chest_range', 'TEMP_chest_range', 'EDA_wrist_range', 'TEMP_wrist_range']]

    # Delete unnecessary columns
    del df_mean['Label_mean']
    del df_sd['Label_sd']
    df_mean.drop(columns=['EDA_wrist_mean', 'TEMP_wrist_mean'], inplace=True)

    # Get the indices corresponding to the relevant states
    min_index2 = df_min[df_min['Label']==2].index
    max_index2 = df_max[df_max['Label']==2].index
    min_index3 = df_min[df_min['Label']==3].index
    max_index3 = df_max[df_max['Label']==3].index
    indices_2 = min_index2.intersection(max_index2)
    indices_3 = min_index3.intersection(max_index3)

    full_df2 = pd.concat([df_mean.iloc[indices_2], df_sd.iloc[indices_2], df_range.iloc[indices_2], df_slope.iloc[indices_2]], axis=1)
    full_df3 = pd.concat([df_mean.iloc[indices_3], df_sd.iloc[indices_3], df_range.iloc[indices_3], df_slope.iloc[indices_3]], axis=1)

    full_df2['Label'] = 2
    full_df3['Label'] = 3

    full_df = pd.concat([full_df2, full_df3], axis=0)

    return full_df


def run_get_pickle_data():
    window_size = 700*5

    subjects = ["S2","S11","S13","S14","S15","S16","S17","S10","S3","S4","S5","S6","S7","S8","S9"]
    for subject in subjects:
        path = "/hpc/group/sta440-f20/WESAD/WESAD/" + subject + "/" + subject +  ".pkl"
        df = get_pickle_data(path)
        df = calculate_window_stats(df, window_size)
        df.to_csv("/work/jmz15/WESAD/data/" + subject  + "_full.csv") # write to csv - TODO: if youre a different user PLEASE CHANGE THE FILE PATH

if __name__ == '__main__':
    run_get_pickle_data()
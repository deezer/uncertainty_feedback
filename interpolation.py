import pandas as pd
import numpy as np
from scipy.stats import beta
from scipy.interpolate import LinearNDInterpolator
import pickle

def compute_interpolation(time_bins = 20, a_prior = 500,  b_prior = 500, dataset = 'deezer' ):
        
        #pre-processing data
        ###################################
        #data path
        data_path  = data_path  = "data/streams.csv"
        df_all = pd.read_csv(data_path) 


        df_all.sort_values(by = 'ts_listen', inplace = True)
        df_all.reset_index(drop = True, inplace = True)


        #we derive the interpolation from only the train data
        df_all['delta_t'] =  df_all['ts_listen'] - df_all['ts_listen'].min()
        
        #training (first 70\% of the total time window) validation (from 70\% to 85\%) and test (the rest of the data). 
        val = df_all.delta_t.max()*0.7
        test = df_all.delta_t.max()*0.85

        df_all['set'] = 'train'
        df_all.loc[df_all.delta_t >= val, 'set'] = 'test'
        df_all.loc[df_all.delta_t >= test, 'set'] = 'test'
        
        df = df_all[df_all.set == 'train'].copy()

        #add time since last listen, or recency
        df['end_song'] = df['ts_listen'] 
        df['last_y_eq_1'] = df['end_song'].where(df['y'] == 1)

        # group by ['user_id', 'track_id'] and forward-fill the last timestamp where y == 1
        df['last_y_eq_1'] = df.groupby(['user_id', 'track_id'])['last_y_eq_1'].ffill()

        # shift the last_y_eq_1 by one period within each group to get the previous timestamp
        df['last_y_eq_1'] = df.groupby(['user_id', 'track_id'])['last_y_eq_1'].shift()

        # get the time difference
        df['time_diff'] = df['ts_listen'] - df['last_y_eq_1']

        #the max rep for Last.fm is 73, for the Deezer dataset it's 57
        max_rep = 73
        
        if dataset == 'deezer':
                #here we consider the behavior for time differences superior to 134s, which is the lower end of the track duration in our data
                df = df[(df['time_diff'] >= 134) | (df['time_diff'].isna())]
                max_rep = 57
       

        print('data size: ',len(df))
        print(df[['user_id','track_id']].nunique())

        #add variables such as time difference or number of LEs
        df.sort_values(by=['ts_listen'], inplace = True)
        df.reset_index(inplace = True, drop = True)
        df['n_listen'] = df.groupby(['user_id','track_id'])['y'].transform('cumsum')
        df['listen_count'] = df.groupby(['user_id','track_id'])['n_listen'].shift(periods=1, fill_value=0)

        #get the max number of play counts per pair user-item
        temp = df.groupby(['user_id','track_id'])['y'].sum().reset_index(name = 'total_rep')
        df = df.merge(temp, on = ['user_id','track_id'])


        #get the max number of interactions per pair user-item
        temp = df.groupby(['user_id','track_id'])['y'].count().reset_index(name = 'total_int')
        df = df.merge(temp, on = ['user_id','track_id'])

        #binarize the time between LEs
        df['time_diff_h'] = df['time_diff']/3600
        df['log_time'] = df['time_diff_h'].apply(np.log10)

        #the maximum number of playcount should change with the dataset, for Last.fm we use <=73
        
        df1 = df[(df.time_diff >0)&(df.total_rep <=max_rep)].copy()
        time_data = df1['time_diff']
        # create logarithmically spaced bins
        log_bins = np.logspace(np.log10(time_data.min()), np.log10(time_data.max()), time_bins)
        df1['diff_range'] = pd.cut(time_data, bins=log_bins, labels=False, include_lowest=True)

        #uncertainty computing
        ###################################
        #get the number of listens and skips for computing the beta posteriors
        temp1 = df1[df1.y == 1].groupby(['listen_count','diff_range'])['ts_listen'].count().reset_index(name = 'successes')
        temp2 = df1[df1.y == 0].groupby(['listen_count','diff_range'])['ts_listen'].count().reset_index(name = 'failures')
        temp3 = df1.groupby('diff_range')['time_diff'].mean().reset_index(name = 'median_diff')


        data = temp1.merge(temp2, on = ['listen_count','diff_range'])
        data = data.merge(temp3, on = 'diff_range')

        data['total'] = data['successes'] + data['failures']

        playcount = []
        time_diff = []
        means = []
        stds = []
        upper = []
        lower= []


        #compute the posterior distribution
        for index, row in data.iterrows():
                a = a_prior + row['successes']
                b = b_prior + row['failures']

                playcount.append(row['listen_count'])
                time_diff.append(row['median_diff'])
                means.append(beta.mean(a, b, loc=0, scale=1))
                stds.append(beta.std(a, b, loc=0, scale=1))
                lower.append(beta.interval(0.95, a, b, loc=0, scale=1)[0])
                upper.append(beta.interval(0.95, a, b, loc=0, scale=1)[1])
                
    
        data = pd.DataFrame({'listen_count': playcount, 'time_diff':time_diff, 'listen_prob':means, 'upper':upper,'lower':lower, 'sf': sf})
        data['CIw'] = data['upper'] - data['lower']

        data.dropna(inplace = True)

        #compute interpolations
        x = data['listen_count'].values
        y = np.log10(data['time_diff'].values)
        z = data['listen_prob'].values
        
        mean_interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value= a_prior/(a_prior+b_prior))#0.5)

        x = data['listen_count'].values
        y = np.log10(data['time_diff'].values)
        z = data['CIw'].values#data['sf'].values

        cdi_interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value= data.CIw.max())

        #return interpolation
        return mean_interp, cdi_interp

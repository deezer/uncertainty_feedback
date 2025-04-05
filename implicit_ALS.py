import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
import argparse
from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid
from scipy.sparse import csr_matrix
import pickle
import random
import itertools
import pdb
import implicit

np.random.seed(42)

import threadpoolctl
threadpoolctl.threadpool_limits(1, "blas")

#function to generate the beta posterior interpolations
def load_or_generate_interpolation(time_bins, prior, dataset):
  mean_path = f"beta_interpolation/{time_bins}_prior{prior}_mean_inter.pkl"
  cdi_path = f"beta_interpolation/{time_bins}_prior{prior}_cdi_inter.pkl"
    
  if os.path.exists(mean_path) and os.path.exists(cdi_path):
      with open(mean_path, "rb") as file:
          mean_inter = pickle.load(file)
      with open(cdi_path, "rb") as file:
          cdi_inter = pickle.load(file)
  else:
      print(f"Generating interpolation for time_bins={time_bins}, a_prior={prior}")
        mean_inter, cdi_inter = compute_interpolation(time_bins,prior, prior, dataset)

      # Save the interpolation results
      with open(mean_path, "wb") as file:
          pickle.dump(mean_inter, file)
      with open(cdi_path, "wb") as file:
          pickle.dump(cdi_inter, file)

  return mean_inter, cdi_inter

#function to compute Recall, we assume that 2 items are in the test set
def compute_recall(df, k):
    temp = df[df.ranking <= k].copy()
    temp = temp.groupby('user_index')['y'].sum().reset_index(name = 'num_true_positives')
    temp['recall'] = temp['num_true_positives']/2.
    recall = temp['recall'].values
    return np.mean(recall)

#function to compute NDCG, we assume that 2 items are in the test set
def compute_ndcg(df,k):
    temp = df[df.ranking <= k].copy()
    temp['weight'] = 1/(np.log2(1 + temp['ranking']))
    rank = np.arange(1,3)
    idcg = np.sum(1/np.log2(1+rank))

    temp['gain'] = (temp['weight']*temp['y'])
    temp = temp.groupby('user_index')['gain'].sum().reset_index(name = 'cum_gain')
    temp['ndcg'] = temp['cum_gain']/idcg
    return np.mean(temp['ndcg'].values)#np.mean(means), np.std(means)


#define dataset
dataset = 'deezer'
#data need to be preprocessed with a column 'set' with values {'train','validation','test'}
data_path  = "/data/data_ALS.csv"
#define weight scheme can be, for the baselines:'linear', 'logarithmic', 'WNeuMF'
#using uncertainty: 'uncertainty_sum_posterior', 'uncertainty_log_sum_posterior and 'uncertainty_confidence'
weight = 'uncertainty_sum_posterior'
#define test
mode = 'test' 
use_gpu = True

#load posteriror interpolation
time_bins = 100
prior = 100


mean_inter, cdi_inter = load_or_generate_interpolation(time_bins, a_prior, dataset)

def PositionalMapping(ids):
    return {id: i for i, id in enumerate(ids)}

#load data
df = pd.read_csv(data_path) 

#split data into train validation and test 
data = df[(df.set == 'train')&(df.y == 1)].copy()
df_val = df[df.set == 'validation'].copy()
df_test = df[df.set == 'test'].copy()

user_mapping = PositionalMapping(data.userId.unique())
item_mapping = PositionalMapping(data.itemId.unique())

data['user_index'] = data['userId'].apply(lambda x: user_mapping[x] )
data['item_index'] = data['itemId'].apply(lambda x: item_mapping[x] )

alphas = [0.1, 0.5, 1, 1.5, 2, 10, 40, 100]
dims = [32]
regs = [0.01, 0.05, 0.1, 0.5, 0.7, 1, 1.5] 
epsilons =[0.1, 0.5, 0.8, 1, 1.5]

configurations = list(itertools.product(alphas, epsilons, dims, regs))

al = []
ep =[]
dim = []
reg = []

recall10 = []
recall20 = [] 
ndcg10 = []
ndcg20 = []

if 'uncertainty' in weight:
  
    data['end_song'] = data['ts_listen'] 
    data['last_y_eq_1'] = data['end_song']
    data['last_y_eq_1'] = data.groupby(['user_index', 'item_index'])['last_y_eq_1'].ffill()
    data['last_y_eq_1'] =data.groupby(['user_index', 'item_index'])['last_y_eq_1'].shift()
    data['time_diff'] = data['ts_listen'] - data['last_y_eq_1']
   
    data.sort_values(by=['ts_listen'], inplace = True)
    data.reset_index(inplace = True, drop = True)
    data['n_listen'] = data.groupby(['user_index','item_index'])['y'].transform('cumsum')
    data['listen_count'] = data.groupby(['user_index','item_index'])['n_listen'].shift(periods=1, fill_value=0)
    data['log_diff'] = data['time_diff'].apply(np.log10)
    x = data[['listen_count','log_diff']].values

    mean_pi = mean_inter([x])
    cdi = cdi_inter([x])
    data['mean_pi'] = mean_pi[0]
    data['cdi'] = cdi[0] 
    
else:
    data = data.groupby(['user_index','item_index'])['y'].sum().reset_index(name = 'int_count')

dfs = []

if mode == 'validation':
    test_data = df_val.copy()
else:
    test_data = df_test.copy()

        
test_data['user_index'] = test_data['userId'].apply(lambda x: user_mapping[x] )
test_data['item_index'] = test_data['itemId'].apply(lambda x: item_mapping[x] )
 
if weight == 'linear':
    df = data.copy()
    df['weight'] = df['int_count']

#parameter loop
for alpha, epsilon, n_dim, regu in configurations:
    print('alpha: ', alpha, 'epsilon :', epsilon, 'n_dim: ', n_dim, 'reg: ', regu)

    if weight  == 'uncertainty_sum_posterior':
        data['confidence'] =  data['mean_pi'] 
        df = data.groupby(['user_index','item_index'])['confidence'].sum().reset_index(name = 'weight')
    
    if weight  == 'uncertainty_log_sum_posterior':
        data['confidence'] =  data['mean_pi']
        df = data.groupby(['user_index','item_index'])['confidence'].sum().reset_index(name = 'weight')
        
        df['weight'] = 1+ df['weight']
        df['weight'] = df['weight'].apply(np.log)

    if weight  == 'uncertainty_confidence':
        data['confidence'] =  data['mean_pi']/(epsilon + data['cdi']) 
        df = data.groupby(['user_index','item_index'])['confidence'].sum().reset_index(name = 'weight')
        
    if weight == 'logarithmic':        
        df = data.copy()
        df['weight'] = 1+ df['int_count']/epsilon
        df['weight'] = df['weight'].apply(np.log)
            

    if weight == 'WNeuMF':
        df = data.copy()
        temp = df.groupby('item_index')['int_count'].mean().reset_index(name='ri')
        df = df.merge(temp, on='item_index', how='left')
        df['rui'] = df['int_count']/df['ri']     
        df['weight'] = np.log(1 + df['rui'] / epsilon)
            
    
    for col in ['user_index', 'item_index']:
        df[col] = df[col].astype('int32')
    df['weight'] = df['weight'].astype('float32')



    user_play = csr_matrix((df.weight.values,(df.user_index.values, df.item_index.values)),dtype=np.float32)

    for i in np.arange(0,10):

        model = AlternatingLeastSquares(factors=n_dim, regularization=regu, alpha= alpha,iterations=100, use_gpu=use_gpu, calculate_training_loss= False)#dtype = np.float32, dtype = np.float32)

      #initialize both the GPU and CPU versions of ALS in the same way
        user_factors_base = np.random.uniform(low=0, high=1, size=(user_play.shape[0], model.factors))*0.01
        item_factors_base = np.random.uniform(low=0, high=1, size=(user_play.shape[1], model.factors))*0.01

        if use_gpu:
            user_factors_base= implicit.gpu.Matrix(user_factors_base.astype(np.float32))
            item_factors_base = implicit.gpu.Matrix(item_factors_base.astype(np.float32))
        
        model.user_factors = user_factors_base
        model.item_factors = item_factors_base
        
        model.fit(user_play*alpha,  show_progress = False)
        
        
        #compute the scores for the users in the test set 
        user_ids = test_data.user_index.unique().astype('int32')
        ids, scores = model.recommend(user_ids, user_play[user_ids], N=200, filter_already_liked_items=True)
        user_ids_repeated = np.repeat(user_ids, scores.shape[1])

        # Flatten the item_id and scores array
        flattened_items = ids.flatten()
        flattened_scores = scores.flatten()

        # Create a DataFrame with 'user_id' and 'score' columns
        df_rec = pd.DataFrame({'user_index': user_ids_repeated,'item_index':flattened_items, 'score': flattened_scores})

        df_rec['ranking'] = df_rec.groupby('user_index')['score'].rank(method='first', ascending=False)

        df_rec = df_rec.merge(test_data[['user_index','item_index','y']], how = 'left')
        df_rec['y'] = df_rec['y'].fillna(0)

        al.append(alpha)
        ep.append(epsilon)
        dim.append(n_dim)
        reg.append(regu)

        recall10.append(compute_recall(df_rec, 10))
        recall20.append(compute_recall(df_rec, 20))

        ndcg10.append(compute_ndcg(df_rec, 10))
        ndcg20.append(compute_ndcg(df_rec, 20))
        print(np.mean(recall10))

results = pd.DataFrame({'alpha':al, 'epsilon':ep,#'reg': reg,#'n_dim': dim,
                            'recall10': recall10,
                            'recall20': recall20,
                            'ndcg10': ndcg10,
                            'ndcg20': ndcg20,
                            })

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 0)

#print the results 
grouped_results = results.groupby(['alpha','epsilon']).agg(['mean', 'std']).map(lambda x: f"{x:.5f}")
print(grouped_results)

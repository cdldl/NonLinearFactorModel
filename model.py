######
# Non linear factor model
######
######
# Potential Improvements
# Use sqrt(volume) as proxy for market capitalization for benchmark
# TS Factor standardization
# Garch Return Standardization
# AutoML Cross-CV for predictions
# Rocket: Weighted by probabilities instead of equal weighted
# Autoencoder: if number of variables was many
# Compare to Benchmark
# Parallelize the program
# Use numba for functions
# Modularize
# Requirements.txt file
######

import os
import numpy as np
import pandas as pd
import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor

buckets = 10

input_path = "C:/Users/cyril/Desktop/Coding/kernelcapital/ORATS/data/data.csv"
input_path = "/home/cyril/data/data.csv"

data = pd.read_csv(input_path)
data['log_last'] = np.log(data['last'])
data['ret'] = data.groupby('ticker').log_last.diff()
data['forret'] = data.groupby('ticker').ret.shift(-1)
data['rev'] = - data.ret
data = data.dropna()
data['mom'] = data.groupby('ticker').ret.rolling(21).sum().fillna(0).values
data['vol'] = data.groupby('ticker').ret.rolling(50).std().fillna(0).values
data['sqrt_volume'] = np.sqrt(1 + data.volume)
data['weights'] = data.groupby('date').sqrt_volume.apply(lambda x: x / x.sum()).values

benchmark = data.groupby('date').apply(lambda x: np.sum(x['weights']* x['ret']))

exposure_vars = ['rev','mom','vol']
def zScore(i):
    i = i.values
    meanExp = np.mean(i)
    sigmaExp = np.std(i)
    sigmaEWMA = np.zeros(len(i))
    ts = (i - meanExp)**2
    var_past2 = sigmaExp**2
    for j in range(len(i)):
        var_past2 = 0.1 * ts[j] + 0.9 * var_past2
        sigmaEWMA[j] = var_past2
    sigmaEWMA[np.where(sigmaEWMA == 0)[0]] = 1
    return (i - meanExp) / sigmaEWMA

for i in exposure_vars:
    data[i] = np.concatenate(data.groupby('ticker')[i].apply(zScore).values)


def standardizeReturns(forret):
    forret = forret.values
    alpha = 0.1
    beta = 0.81
    sdReturns = np.std(forret)
    ts = forret**2
    sigma_garch = np.zeros(len(forret))
    for i in range(len(forret)):
        if i == 1:
            sigma_garch[i] = (1 - alpha - beta) * sdReturns**2 + alpha * ts[i]
        else:
            sigma_garch[i] = (1 - alpha - beta) * sdReturns**2 + alpha * ts[i] + \
                            beta * sigma_garch[i-1]
    sigma_garch = np.sqrt(sigma_garch)
    return sigma_garch

data['sigmaGarch'] = np.concatenate(data.groupby('ticker')['forret'].apply(standardizeReturns).values)
data['forret_std'] = data.forret / data.sigmaGarch

data = data.dropna()
benchmark = benchmark.reset_index()
max_lookback= 1
exposure_vars.append('forret_std')
for i in range(2, (len(np.unique(data.date)) -1)):
    print(i / len(np.unique(data.date)))
    start_date = np.sort(np.unique(data.date))[i - max_lookback]
    end_date = np.sort(np.unique(data.date))[i]
    tmp_data = data[(data.date > start_date) & (data.date <= end_date)]
    to_predict = data[data.date == np.sort(np.unique(data.date))[i+1]]
    train_data = TabularDataset(tmp_data[exposure_vars])
    test_data = TabularDataset(to_predict[exposure_vars])
    savedir = f'saved_models/'  # where to save trained models
    predictor = TabularPredictor(label='forret_std',
                                 path=savedir,
                                eval_metric='mean_squared_error')
    predictor.fit(train_data=train_data,
                         time_limit=6,
                         ag_args_fit={'num_gpus': 7},
                         num_stack_levels=0,
                         # auto_stack=True,
                         num_bag_folds=3,
                         verbosity=4,
                         #random_seed=0,
                         save_space=True,
                         keep_only_best=True,
                         hyperparameters='default')
    preds = predictor.predict(test_data)
    data.loc[(data.date == np.sort(np.unique(data.date))[i+1]),'preds'] = preds

data = data.dropna()
data['signed_preds'] = np.sign(data.preds)
#data['sigmaGarch_lag']= data.groupby('ticker').sigmaGarch.shift(1).values
data['real_preds'] = data.preds #* data.sigmaGarch_lag

data  =data.dropna()
def fractile(preds,buckets):
    return pd.qcut(preds, np.linspace(0, 1, buckets + 1), labels=False)

data['Basket'] = data.groupby('date').real_preds.apply(fractile, buckets).values


data['post_weights'] = data.groupby(['date','Basket']).real_preds.apply(lambda x: abs(x) / abs(x).sum())

data['realret'] = data.post_weights * data.forret
data = data.dropna()

Perf = data.pivot_table(index=['date'],values=['realret'], columns=['Basket'], aggfunc=np.sum).fillna(0)
Perf['LS'] = Perf.iloc[:, -1] - Perf.iloc[:, 0]

pnl = np.sum(Perf.LS)
sharpe = (np.mean(Perf.LS) * np.sqrt(252)) / np.std(Perf.LS)
hit_ratio = len(np.where(Perf.LS > 0)[0]) / len(Perf.LS)
drawdown = max(Perf.LS.cummax())
profit_ratio = np.sum(Perf.LS)/abs(drawdown)

print(f'pnl {pnl} sharpe {sharpe} hit {hit_ratio} profit_ratio {profit_ratio}')
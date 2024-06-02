# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = '../data/sessionized.csv' # path to input dataset
PATH_TO_PROCESSED_DATA = '../data/sessionized_GRU4Rec' # path where preprocessed dataset will be stored

data = pd.read_csv(PATH_TO_ORIGINAL_DATA, sep=',', usecols=['itemId', 'timestamp', 'session_id'])

# renaming the cols so they fit with the rest of the code
data = data.rename(columns={'itemId': 'ItemId', 'timestamp': 'Time', 'session_id': 'SessionId'})

data['ItemId'] = data['ItemId'].astype(np.int32)
data['Time'] = data['Time'].astype(np.int64)

#data.columns = ['SessionId', 'TimeStr', 'ItemId']
#data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
#del(data['TimeStr'])

#session_lengths = data.groupby('SessionId').size() # count interactions per session
#data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)] -> I already do this in my create_sessionized_dataset.ipynb

# filters dataframe to only contain items with >= 5 interactions
item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]

#session_lengths = data.groupby('SessionId').size()
#data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)] -> I already do this in my create_sessionized_dataset.ipynb

tmax = data.Time.max() # calculates max timestamp in the whole dataset
session_max_times = data.groupby('SessionId').Time.max() # calculates max timestamp for each session
session_train = session_max_times[session_max_times < tmax-86400].index # all sessions not on the last day (24hr = 86400s) are used for training
session_test = session_max_times[session_max_times >= tmax-86400].index # last 24hr are used for testing
#train = data[np.in1d(data.SessionId, session_train)]
#test = data[np.in1d(data.SessionId, session_test)]

train = data[data['SessionId'].isin(session_train)]
test = data[data['SessionId'].isin(session_test)]

#test = test[np.in1d(test.ItemId, train.ItemId)] # filter test set to contain only items present in the training set
test = test[test['ItemId'].isin(train['ItemId'])]

#tslength = test.groupby('SessionId').size()
#test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)] -> I already do this in my create_sessionized_dataset.ipynb

# print summary and save files
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + '_train.csv', sep=',', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + '_test.csv', sep=',', index=False)

tmax = train.Time.max() # calculate max timestamp of training set
session_max_times = train.groupby('SessionId').Time.max() # calculate max timestamp per session in training set

# sessions of last day being picked as val set
session_train = session_max_times[session_max_times < tmax-86400].index
session_valid = session_max_times[session_max_times >= tmax-86400].index

train_tr = data[data['SessionId'].isin(session_train)]
valid = train[train['SessionId'].isin(session_valid)]

#valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)] # validation set should only contain items which are also present in the training set

valid = valid[valid['ItemId'].isin(train_tr['ItemId'])]

#tslength = valid.groupby('SessionId').size()
#valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)] -> I already do this in my create_sessionized_dataset.ipynb

# print summary and save files
print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
train_tr.to_csv(PATH_TO_PROCESSED_DATA + '_train_optim.csv', sep=',', index=False)
print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
valid.to_csv(PATH_TO_PROCESSED_DATA + '_valid.csv', sep=',', index=False)

print(train.head())
print(test.head())
print(train_tr.head())
print(valid.head())


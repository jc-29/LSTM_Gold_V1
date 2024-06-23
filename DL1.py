### This project attempts to use Deep Learning in the form of a LSTM model to predict if the next trading day will be a bullish or bearish day.

import requests
from datetime import date, timedelta
import json
import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import ta_py as ta

current_date = date.today()

daily_start_date = current_date - timedelta(days=367) # 1 year
daily_end_date = current_date - timedelta(days=1)

hourly_start_date = current_date - timedelta(days=31) # 1 month
hourly_end_date = daily_end_date

minute_start_date = current_date - timedelta(days = 3) # 2 days
minute_end_date = daily_end_date

daily_url = f"https://marketdata.tradermade.com/api/v1/timeseries?api_key=aWbM1qDWUTAMn6beRiL_&currency=XAUUSD&format=records&start_date={daily_start_date}&end_date={daily_end_date}&interval=daily&period=1"

# hourly_url = f"https://marketdata.tradermade.com/api/v1/timeseries?api_key=aWbM1qDWUTAMn6beRiL_&currency=XAUUSD&format=records&start_date={hourly_start_date}&end_date={hourly_end_date}&interval=hourly&period=1"

# minute_url = f"https://marketdata.tradermade.com/api/v1/timeseries?api_key=aWbM1qDWUTAMn6beRiL_&currency=XAUUSD&format=records&start_date={minute_start_date}&end_date={minute_end_date}&interval=minute&period=1"

# daily_records = json.loads(requests.get(daily_url).text)['quotes']
# daily_records = pd.DataFrame(daily_records).dropna()
# daily_records.to_csv('C:/Users/User/Desktop/Trading Research/program/gold_data.csv', index=False)
daily_records = pd.read_csv('C:/Users/User/Desktop/Trading Research/program/gold_data.csv')
# hourly_records = json.loads(requests.get(hourly_url).text)
# minute_records = json.loads(requests.get(minute_url).text)

daily_records['open_close_diff'] = daily_records['close'] - daily_records['open']
daily_records['high_low_diff'] = daily_records['high'] - daily_records['low']
sma_period = 20
sma_records = ta.sma(daily_records['close'], sma_period)
rsi_records = ta.rsi(daily_records['close'], sma_period)
daily_records = daily_records[sma_period - 1:].reset_index(drop=True)
daily_records['sma'] = sma_records
daily_records['rsi'] = rsi_records
daily_records['label'] = (daily_records['open'] < daily_records['close']).astype(int) # 0 for bearish day, 1 for bullish day
daily_arr = np.array(daily_records)
# daily_records.to_csv('C:/Users/User/Desktop/Trading Research/program/gold_data_check.csv')

look_back = 100
num_features = 8
windowed_features_arr = []
labels = []
for i in range(look_back, len(daily_records)):
    record_set = [[] for _ in range(num_features)]
    labels.append(daily_records.loc[i]['label'])
    for j in range(i - look_back, i):
        record = daily_records.loc[j]
        record_set[0].append(record['open'])
        record_set[1].append(record['high'])
        record_set[2].append(record['low'])
        record_set[3].append(record['close'])
        record_set[4].append(record['open_close_diff'])
        record_set[5].append(record['high_low_diff'])
        record_set[6].append(record['sma'])
        record_set[7].append(record['rsi'])
    windowed_features_arr.append(record_set)
    
windowed_features_arr = np.array(windowed_features_arr)
labels = np.array(labels)
    
n = len(windowed_features_arr)
train_data = windowed_features_arr[:int(0.5 * n)] # 50% of data
train_labels = labels[:int(0.5 * n)]
test_data = windowed_features_arr[int(0.5 * n) : int(0.75 * n)] # 25% of data
test_labels = labels[int(0.5 * n) : int(0.75 * n)]
validation_data = windowed_features_arr[int(0.75 * n):] # 25% of data
validation_labels = labels[int(0.75 * n):]

################################################################################################
model = keras.models.Sequential()
model.add(keras.layers.InputLayer((num_features, look_back)))
model.add(keras.layers.LSTM(150)) # better performances were observed with low number of neurons (e.g. 8)
model.add(keras.layers.Dense(units = 1, activation='sigmoid'))
model.summary()

optimizer = keras.optimizers.Adadelta(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mae", metrics=["accuracy"])
model.fit(train_data, train_labels, validation_data = (validation_data, validation_labels), epochs = 50, batch_size=20)
model.evaluate(test_data, test_labels)

correct_count = 0
incorrect_count = 0
total_count = len(test_labels)
data = [test_data[:10]]
lbls = test_labels[:10]
# for i in range(10):
#     lbls.append(test_labels[i])
#     data.append(test_data[i])

print(model.predict(data))
print(lbls)



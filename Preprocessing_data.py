#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:11:58 2018

@author: ryokugyu
"""

# load and clean-up data
import numpy as np
import pandas as pd

# fill missing values with a value at the same time one day ago
def filling_missing_values(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if np.isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]

# load all data into workspace
dataset = pd.read_csv('/home/ryokugyu/Downloads/electricity_consumption/household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# marked missing values
dataset.replace('?', np.nan, inplace=True)
# make
dataset = dataset.astype('float32')

# fill missing
filling_missing_values(dataset.values)

#column added for total remainder readings
values = dataset.values
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
# save updated dataset
dataset.to_csv('/home/ryokugyu/Downloads/electricity_consumption/household_power_consumption.csv')

# resample minute data to total for each day
from pandas import read_csv

# load the new file
dataset = read_csv('/home/ryokugyu/Downloads/electricity_consumption/household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# resample data to daily from minutes
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()

print(daily_data.shape)
print(daily_data.head())
# save
daily_data.to_csv('/home/ryokugyu/Downloads/electricity_consumption/household_power_consumption.csv')
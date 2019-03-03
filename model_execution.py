#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 2:11:58 2018

@author: ryokugyu
"""

import math as math
import numpy as np
import pandas as pd
import sklearn 
import sklearn.linear_model as linear_model 
from sklearn.pipeline import Pipeline
import matplotlib as mtplt

def dividing_dataset(data):
	# split into standard weeks
	training_data, test_data = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	training_data = np.array(np.split(training_data, len(training_data)/7))
	test_data =np.array(np.split(test_data, len(test_data)/7))
	return training_data, test_data

def forecast_evaluation(real_value, value_got):
	scores = list()
	for i in range(real_value.shape[1]):
		mean_square_error= sklearn.metrics.mean_squared_error(real_value[:, i], value_got[:, i])
		root_mean_square_error = math.sqrt(mean_square_error)
		scores.append(root_mean_square_error)
	s = 0
    
	for row in range(real_value.shape[0]):
		for col in range(real_value.shape[1]):
			s += (real_value[row, col] - value_got[row, col])**2
	score = math.sqrt(s / (real_value.shape[0] * real_value.shape[1]))
	return score, scores

def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

def acquire_models(models=dict()):
	models['Linear regression'] = sklearn.linear_model.LinearRegression()
	models['Lasso'] = sklearn.linear_model.Lasso()
	models['Ridge'] = sklearn.linear_model.Ridge()
	models['Elastic Net'] = sklearn.linear_model.ElasticNet()
	models['Huber Regressor'] = sklearn.linear_model.HuberRegressor()
	models['Lars'] = sklearn.linear_model.Lars()
	models['LassoLars'] = sklearn.linear_model.LassoLars()
	print('Defined %d models' % len(models))
	return models

def make_pipeline(model):
	steps = list()
	steps.append(('standardize', sklearn.preprocessing.StandardScaler()))
	steps.append(('normalize', sklearn.preprocessing.MinMaxScaler()))
	steps.append(('model', model))
	pipeline = Pipeline(steps=steps)
	return pipeline

def forecast(model, input_x, n_input):
	yhat_sequence = list()
	input_data = [x for x in input_x]
	for j in range(7):
		X = np.array(input_data[-n_input:]).reshape(1, n_input)
		yhat = model.predict(X)[0]
		yhat_sequence.append(yhat)
		input_data.append(yhat)
	return yhat_sequence

#single series of total daily power consumed each week
def multiple_to_single(data):
	total_power_week_series = [week[:, 0] for week in data]
	total_power_week_series = np.array(total_power_week_series).flatten()
	return total_power_week_series

def supervise(history, prior_days):
	data = multiple_to_single(history)
	X, y = list(), list()
	ix_start = 0
	for i in range(len(data)):
		ix_end = ix_start + prior_days
		if ix_end < len(data):
			X.append(data[ix_start:ix_end])
			y.append(data[ix_end])
		# move along one time step
		ix_start += 1
	return np.array(X), np.array(y)

# fit a model and make a forecast
def sklearn_predict(model, history, prior_days):
	# prepare data
	train_x, train_y = supervise(history, prior_days)
	# make pipeline
	pipeline = make_pipeline(model)
	# fit the model
	pipeline.fit(train_x, train_y)
	# predict the week, recursively
	yhat_sequence = forecast(pipeline, train_x[-1, :], prior_days)
	return yhat_sequence

def model_evaluation(model, train, test, prior_days):
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		yhat_sequence = sklearn_predict(model, history, prior_days)
		predictions.append(yhat_sequence)
		history.append(test[i, :])
	predictions = np.array(predictions)
	score, scores = forecast_evaluation(test[:, :, 0], predictions)
	return score, scores

# load the new file
dataset = pd.read_csv('/home/ryokugyu/Downloads/electricity_consumption/household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
training_data, test_data = dividing_dataset(dataset.values)
# prepare the models to evaluate
models = acquire_models()
prior_days = 7
# evaluate each model
days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday']
for name, model in models.items():
	# evaluate and get scores
	score, scores = model_evaluation(model, training_data, test_data, prior_days)
	# summarize scores
	summarize_scores(name, score, scores)
	# plot scores
	mtplt.pyplot.plot(days, scores, marker='o', label=name)
    

mtplt.pyplot.legend()
mtplt.pyplot.show()
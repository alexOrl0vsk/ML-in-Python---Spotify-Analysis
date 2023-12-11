import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from sklearn import svm
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import(LinearRegression,
								 LogisticRegression)
from sklearn.metrics import(mean_squared_error,
							mean_absolute_error,
							accuracy_score)

from sklearn.model_selection import(train_test_split,
									cross_val_score,
									cross_validate,
									GridSearchCV,
									RandomizedSearchCV)

from sklearn.neural_network import(MLPRegressor,
								   MLPClassifier)

from sklearn.preprocessing import(StandardScaler,
							      MinMaxScaler,
							      RobustScaler, 
							      OneHotEncoder,
							      PolynomialFeatures)

from sklearn.compose import make_column_transformer

from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency


def loadData(raw):
	# input:  raw (bool) :: True -> original data, 
	#						False -> combine playlist/chart data 
	df = pd.read_csv('spotify2023.csv', encoding="ISO-8859-1")
	# convert stream count to 'millions of streams' for easier visualization 
	df['streams'] = pd.to_numeric(df['streams'], errors='coerce')/1000000
	df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors='coerce')
	df['in_shazam_charts'] = pd.to_numeric(df['in_shazam_charts'], errors='coerce')
	# Filter the rows where 'track_title' is not equal to "Love Grows (Where My Rosemary Goes)"
	df = df[df['track_name'] != "Love Grows (Where My Rosemary Goes)"]
	# Replace string Nan's with NumPy NaN's
	df.replace('NaN',np.nan)
	# Fill Nans with zeros instead
	df.fillna(0)
	if raw != True: 
		# Columns to sum
		playlists = ['in_deezer_playlists', 'in_spotify_playlists', 'in_apple_playlists']
		charts = ['in_deezer_charts', 'in_spotify_charts','in_apple_charts','in_shazam_charts']
		# New columns with combined playlist/chart data
		df['playlist_presence'] = df[playlists].sum(axis=1)
		df['chart_presence'] = df[charts].sum(axis=1)
		# Drop old playlist/chart data
		df = df.drop(columns=['in_deezer_playlists','in_spotify_playlists','in_apple_playlists','in_spotify_charts','in_apple_charts','in_shazam_charts','in_deezer_charts'])
	
	# Drop data which is not usefull anyways
	df = df.drop(columns=['artist(s)_name','track_name', 'released_day','released_month']) #,'key','mode'
	return df

def getSplit(X,y):
	X_train, X_test_plus, y_train, y_test_plus = train_test_split(X, y, test_size=0.4 , random_state=11)
	X_valid, X_test, y_valid, y_test = train_test_split(X_test_plus,y_test_plus, test_size = 0.5 , random_state=11)
	return X_train,X_valid,X_test,y_train,y_valid,y_test

def featureCases(caseID):
	df = loadData(False)
	y = df['streams']
	X = df.drop(columns=['streams'])
	match caseID:
		case 0:	# all features 
			encoder = make_column_transformer(
						(OneHotEncoder(), ['key','mode']),
						remainder='passthrough'
						)
			X = pd.DataFrame(encoder.fit_transform(X))
		case 1: # all features WIHOUT categorical variables 
			X = X.drop(columns=['key','mode'])
		case 2: # custom selected feature cases 
			X = df[['artist_count','bpm']]
		case 3:
			X = df[['bpm','valence_%','speechiness_%']]
		case 4: # PCA without categorical, only first 6 components
			X = df.drop(columns=['streams','key','mode'])
			pca = PCA().set_output(transform="pandas").fit(X)
			X = pca.transform(X).iloc[:,:6]
		case 5: # PCA w/ categorical encoded , first 6 component 
			encoder = make_column_transformer(
						(OneHotEncoder(), ['key','mode']),
						remainder='passthrough'
						)
			X_encoded = encoder.fit_transform(X)
			pca = PCA().set_output(transform="pandas").fit(X_encoded)
			X = pca.transform(X_encoded).iloc[:,:6]
		case _: 
			raise ValueError

	return getSplit(X,y)

def singles_or_colabs(Singles):
	# Separate predictions for single releases/ colaborations 

	df = loadData(False)
	X = df.drop(columns=['streams','key','mode'])

	X__singles = X[X['artist_count'] == 1]
	X__colabs = X[X['artist_count'] > 1]

	y__singles = df.loc[df['artist_count'] == 1, 'streams']
	y__colabs = df.loc[df['artist_count'] > 1, 'streams']

	return getSplit(X__singles,y__singles) if Singles else getSplit(X__colabs,y__colabs)

def naiveLoop():
	# Trial and error method for comparing results for different feature selections / scalers / models 
	models = [LinearRegression(),MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=10000, solver='adam'),svm.SVR(C=1)] 
	scalers = [StandardScaler(),MinMaxScaler(),RobustScaler()]

	errors = []

	for caseID in range(0,6):
		for model in models:
			for scaler in scalers:
				RMSE_train, RMSE_valid = customPipe(caseID,scaler,model)
				errors.append(
					{
						'CaseID': caseID,
						'Model' : type(model).__name__,
						'Scaler' : type(scaler).__name__,
						'RMSE_train' : RMSE_train,
						'RMSE_valid' : RMSE_valid
					})

	df_errors = pd.DataFrame(errors)

	print(df_errors.round(3).sort_values(by=['Model','CaseID']))
	df_errors.to_csv("results_csv/naiveLoop.csv")		




	return

def customPipe(caseID,scaler,model):
	# used in naiveLoop() to return errors 
	try:
		X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(caseID)

		myPipe = make_pipeline(scaler,model)
		myPipe.fit(X_train,y_train)

		RMSE_train = mean_squared_error(y_train,myPipe.predict(X_train),squared=False)
		RMSE_valid = mean_squared_error(y_valid,myPipe.predict(X_valid),squared=False)

		return [RMSE_train,RMSE_valid] #{"RMSE_train": RMSE_train,"RMSE_valid":RMSE_valid}

	except ValueError:
		print("Select one of predefined cases!")
	return [np.nan, np.nan]

def trySizesMLP():
	# gridsearch for optimal MLP with RobustScaler() without PCA
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(1)
	scaler = RobustScaler()
	model = MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=20000, solver='adam')
	myPipe = make_pipeline(scaler,model)
	
	param_grid = {
	    'mlpregressor__hidden_layer_sizes': [(20,20,15,10,1), (20,15,10,10,5), (20,15,10,5,1),(20,15,10,5,1), (20,15,10,1), (20,15,15,5),(20,15,15,1),(15,15,10,1), (15,10,10,1),
	    									(30,20,10,1),(20,15,5,1),(15,15,10),(20,20,15),(20,15,10),(20,10,5),(20,15,5)],
	    'mlpregressor__max_iter': [ 10000, 20000],
	    'mlpregressor__activation': ['tanh', 'relu','identity','logistic'],
	    'mlpregressor__solver': ['adam','lbfgs','sgd' ], 
	    'mlpregressor__alpha': [0.0001, 0.0002, 0.0003, 0.0004],
	    'mlpregressor__learning_rate': ['constant','invscaling','adaptive'],
	}

	grid = RandomizedSearchCV(myPipe, param_grid, scoring=['neg_root_mean_squared_error','r2'],
						random_state = 2023, 
						refit = 'neg_root_mean_squared_error',
						n_jobs = -1,
						verbose = 3,
						cv = 5,
						n_iter = 40)

	grid.fit(X_train, y_train)

	print('\nBest params :: \n',grid.best_params_) 

	print('RMSE for train :: ',mean_squared_error(y_train,grid.predict(X_train),squared=False))
	print('RMSE for valid :: ',mean_squared_error(y_valid,grid.predict(X_valid),squared=False))
	print('RMSE for test :: ',mean_squared_error(y_test,grid.predict(X_test),squared=False))

	df_temp = pd.DataFrame(grid.cv_results_)
	df_temp = df_temp.sort_values("rank_test_r2")
	df_temp.to_csv("results_csv/trySizesMLP.csv")

def gridSeach_PCA(caseID, mlp):
	# general pipeline with different (no. of PCA components / models parameters) with same step names 

	# input :: caseID --> feature set
	#		   mlp (bool) --> True uses param_grid_MLP
	#					  --> False uses param_grid_SVR
	# output :: a dataframe with sorted scores

	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(caseID)
	model = MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=10000, solver='adam') if mlp else svm.SVR(kernel='rbf',degree = 4,C = 1000)

	MyPipe = Pipeline([
		("scale", RobustScaler()),
		("PCA", PCA().set_output(transform="pandas")),
		("model", model)
		])

	param_grid_MLP = {
		'PCA__n_components' : np.arange(1,20),
	    'model__hidden_layer_sizes': [(15,15,10),(20,15,10,1),(15,10,10,1)],
	    'model__max_iter': [ 10000, 20000],
	    'model__activation': ['relu','identity','logistic'],
	    'model__alpha': [0.0001, 0.0002, 0.0003]
	}
	param_grid_SVR = {
		'PCA__n_components' : np.arange(1,20),
		'model__C' : np.logspace(-3,3,7),
		'model__degree' : np.arange(1,6),
		'model__kernel' : ['linear','poly','rbf','sigmoid'],
	}

	param_grid = param_grid_MLP if mlp else param_grid_SVR

	grid = RandomizedSearchCV(MyPipe, param_grid, scoring=['neg_root_mean_squared_error','r2'],
		random_state = 1, 
		refit = 'neg_root_mean_squared_error',
		n_jobs = -1,
		verbose = 3,
		cv = 5,
		n_iter = 50)

	grid.fit(X_train, y_train)

	print('\nBest params :: \n',grid.best_params_) 

	print('RMSE for train :: ',mean_squared_error(y_train,grid.predict(X_train),squared=False))
	print('RMSE for valid :: ',mean_squared_error(y_valid,grid.predict(X_valid),squared=False))
	print('RMSE for test :: ',mean_squared_error(y_test,grid.predict(X_test),squared=False))

	df_temp = pd.DataFrame(grid.cv_results_)
	df_temp = df_temp.sort_values("rank_test_r2")
	return df_temp
	

def modelTuning(featureCaseID): 
	# grid search without without taking PCA 
	# change scoring to get best score as RMSE or R2 in the output file
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(featureCaseID)
	model_params = {
		'svr' : {
			'model' : svm.SVR(C=1),
			'params' : {
				'model__C' : np.logspace(-3,3,7),
				'model__degree' : np.arange(1,6),
				'model__kernel' : ['linear','poly','rbf','sigmoid'],
			}
		},
		'mlp' : {
			'model' : MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=10000, solver='adam'),
			'params' : {
				'model__hidden_layer_sizes': [(15,15,10), (20,15,5), (15,10,5),(15,15,10,1),(20,15,10,1)],
			    'model__max_iter': [ 10000, 20000,30000],
			    'model__alpha': [0.0001, 0.0002],
			}
		}
	}

	scores = []

	for model_name, mp in model_params.items():
		MyPipe = Pipeline(steps=[('scale', RobustScaler()),
								('model', mp['model'])
								])
		grid = RandomizedSearchCV(MyPipe, mp['params'], scoring=['neg_root_mean_squared_error','r2'],
								random_state = 15, 
								refit = 'r2',
								n_jobs = -1,
								verbose = 3,
								cv = 5,
								n_iter = 50
								)
		grid.fit(X_train,y_train)
		scores.append({
				'model' : model_name,
				'best_params' : grid.best_params_,
				'best_score' : grid.best_score_,
			})

	pd.DataFrame(scores).to_csv("results_csv/noPCA_tuning.csv")


def bestConfigs(caseID):
	# selected cases with best results as discovered by previous gridsearches  
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(caseID)
	scaler = RobustScaler()
	modelMLP = MLPRegressor(hidden_layer_sizes=(20, 15, 10, 1), random_state=1,max_iter=20000, solver='adam',alpha=0.0001)
	modelSVR = svm.SVR(kernel='rbf',
					degree = 4,
					C = 1000)

	models = [modelMLP, modelSVR]

	bestConfigs = []

	for model in models:
		myPipe = make_pipeline(scaler,model)
		myPipe.fit(X_train,y_train)

		RMSE_train_myPipe = mean_squared_error(y_train,myPipe.predict(X_train),squared=False)
		RMSE_valid_myPipe = mean_squared_error(y_valid,myPipe.predict(X_valid),squared=False)
		RMSE_test_myPipe = mean_squared_error(y_test,myPipe.predict(X_test),squared=False)
		# fit train + validation data 
		myPipe.fit(pd.concat([X_train, X_valid], axis=0),pd.concat([y_train, y_valid], axis=0))
		RMSE_test_myPipe2 = mean_squared_error(y_test,myPipe.predict(X_test),squared=False)
		print(f'\n\n\t{model} without PCA :: ')
		print(f'RMSE for training\t:: {RMSE_train_myPipe}')
		print(f'RMSE for validation\t:: {RMSE_valid_myPipe}')
		print(f'RMSE for test before validation fit\t:: {RMSE_test_myPipe}')
		print(f'RMSE for test after validation fit\t:: {RMSE_test_myPipe2}')


		plt.figure(figsize=(12, 8))
		sns.scatterplot(x=y_test, y=myPipe.predict(X_test), alpha=0.6, edgecolor=None)
		plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
		plt.xlabel('Actual Streams', fontsize=14)
		plt.ylabel('Predicted Streams', fontsize=14)
		plt.title(f'Actual vs. Predicted Streams w/\n{model}', fontsize=16)
		plt.grid(True, which='both', linestyle='--', linewidth=0.5)
		plt.tight_layout()


		pipePCA = make_pipeline(scaler,
								PCA(n_components=4).set_output(transform="pandas"),
								model)
		pipePCA.fit(X_train,y_train)

		RMSE_train_pipePCA = mean_squared_error(y_train,pipePCA.predict(X_train),squared=False)
		RMSE_valid_pipePCA = mean_squared_error(y_valid,pipePCA.predict(X_valid),squared=False)
		RMSE_test_pipePCA = mean_squared_error(y_test,pipePCA.predict(X_test),squared=False)
		# fit train + validation data 
		pipePCA.fit(pd.concat([X_train, X_valid], axis=0),pd.concat([y_train, y_valid], axis=0))
		RMSE_test_pipePCA2 = mean_squared_error(y_test,pipePCA.predict(X_test),squared=False)
		print(f'\n\n\t{model} w/ optimal PCA :: ')
		print(f'RMSE for training\t:: {RMSE_train_pipePCA}')
		print(f'RMSE for validation\t:: {RMSE_valid_pipePCA}')
		print(f'RMSE for test before validation fit\t:: {RMSE_test_pipePCA}')
		print(f'RMSE for test after validation fit\t:: {RMSE_test_pipePCA2}')

		bestConfigs.append(
			{
				'CaseID': caseID,
				'Model' : type(model).__name__,
				'Scaler' : type(scaler).__name__,
				'Without PCA' : {
					'RMSE_train' : RMSE_train_myPipe,
					'RMSE_valid' : RMSE_valid_myPipe,
					'RMSE_test' : RMSE_test_myPipe,
					'RMSE_test (fit on train + validation)' : RMSE_test_myPipe2

				},
				'With PCA' : {
					'RMSE_train' : RMSE_train_pipePCA,
					'RMSE_valid' : RMSE_valid_pipePCA,
					'RMSE_test' : RMSE_test_pipePCA,
					'RMSE_test (fit on train + validation)' : RMSE_test_pipePCA2

				}

			})
		plt.figure(figsize=(12, 8))
		sns.scatterplot(x=y_test, y=pipePCA.predict(X_test), alpha=0.6, edgecolor=None)
		plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
		plt.xlabel('Actual Streams', fontsize=14)
		plt.ylabel('Predicted Streams', fontsize=14)
		plt.title(f'Actual vs. Predicted Streams for\n{model} w/ PCA', fontsize=16)
		plt.grid(True, which='both', linestyle='--', linewidth=0.5)
		plt.tight_layout()


	plt.show()
	pd.DataFrame(bestConfigs).to_csv("results_csv/bestConfigs.csv")

	

	return bestConfigs

def categoricalCorrelation():
	df = loadData(False)
	cross_tab = pd.crosstab(index=df['key'], columns=df['mode'])
	chi2_key = chi2_contingency(cross_tab)
	print("Chi squared p-value for key-mode :: ", chi2_key[1])

def runSinglesvColabs():
	bestConfigs = []

	for Singles in [True,False]:
		X_train,X_valid,X_test,y_train,y_valid,y_test = singles_or_colabs(Singles)
		plot_title = 'Singles only' if Singles else 'Colabs only'
		scaler = RobustScaler()
		modelMLP = MLPRegressor(hidden_layer_sizes=(20, 15, 10, 1), random_state=1,max_iter=10000, solver='adam',alpha=0.0002)
		modelSVR = svm.SVR(kernel='rbf',
						degree = 4,
						C = 1000)

		models = [modelMLP, ]

		print("CASE :: ",plot_title)
		for model in models:
			myPipe = make_pipeline(scaler,model)
			myPipe.fit(X_train,y_train)

			RMSE_train_myPipe = mean_squared_error(y_train,myPipe.predict(X_train),squared=False)
			RMSE_valid_myPipe = mean_squared_error(y_valid,myPipe.predict(X_valid),squared=False)
			RMSE_test_myPipe = mean_squared_error(y_test,myPipe.predict(X_test),squared=False)
			# fit train + validation data 
			myPipe.fit(pd.concat([X_train, X_valid], axis=0),pd.concat([y_train, y_valid], axis=0))
			RMSE_test_myPipe2 = mean_squared_error(y_test,myPipe.predict(X_test),squared=False)
			print(f'\n\n\t{model} without PCA :: ')
			print(f'RMSE for training\t:: {RMSE_train_myPipe}')
			print(f'RMSE for validation\t:: {RMSE_valid_myPipe}')
			print(f'RMSE for test before validation fit\t:: {RMSE_test_myPipe}')
			print(f'RMSE for test after validation fit\t:: {RMSE_test_myPipe2}')

			plt.figure(figsize=(12, 8))
			sns.scatterplot(x=y_test, y=myPipe.predict(X_test), alpha=0.6, edgecolor=None)
			plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
			plt.xlabel('Actual Streams', fontsize=14)
			plt.ylabel('Predicted Streams', fontsize=14)
			plt.title(f'{plot_title} :: Actual vs. Predicted Streams w/\n{model}', fontsize=16)
			plt.grid(True, which='both', linestyle='--', linewidth=0.5)
			plt.tight_layout()

			pipePCA = make_pipeline(scaler,
									PCA(n_components=5).set_output(transform="pandas"),
									model)
			pipePCA.fit(X_train,y_train)

			RMSE_train_pipePCA = mean_squared_error(y_train,pipePCA.predict(X_train),squared=False)
			RMSE_valid_pipePCA = mean_squared_error(y_valid,pipePCA.predict(X_valid),squared=False)
			RMSE_test_pipePCA = mean_squared_error(y_test,pipePCA.predict(X_test),squared=False)
			# fit train + validation data 
			pipePCA.fit(pd.concat([X_train, X_valid], axis=0),pd.concat([y_train, y_valid], axis=0))
			RMSE_test_pipePCA2 = mean_squared_error(y_test,pipePCA.predict(X_test),squared=False)
			print(f'\n\n\t{model} w/ optimal PCA :: ')
			print(f'RMSE for training\t:: {RMSE_train_pipePCA}')
			print(f'RMSE for validation\t:: {RMSE_valid_pipePCA}')
			print(f'RMSE for test before validation fit\t:: {RMSE_test_pipePCA}')
			print(f'RMSE for test after validation fit\t:: {RMSE_test_pipePCA2}')

			bestConfigs.append(
				{
					'Case': "Singles" if Singles else "Colabs",
					'Model' : type(model).__name__,
					'Scaler' : type(scaler).__name__,
					'Without PCA' : {
						'RMSE_train' : RMSE_train_myPipe,
						'RMSE_valid' : RMSE_valid_myPipe,
						'RMSE_test' : RMSE_test_myPipe,
						'RMSE_test (fit on train + validation)' : RMSE_test_myPipe2

					},
					'With PCA' : {
						'RMSE_train' : RMSE_train_pipePCA,
						'RMSE_valid' : RMSE_valid_pipePCA,
						'RMSE_test' : RMSE_test_pipePCA,
						'RMSE_test (fit on train + validation)' : RMSE_test_pipePCA2

					}

				})
			plt.figure(figsize=(12, 8))
			sns.scatterplot(x=y_test, y=pipePCA.predict(X_test), alpha=0.6, edgecolor=None)
			plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
			plt.xlabel('Actual Streams', fontsize=14)
			plt.ylabel('Predicted Streams', fontsize=14)
			plt.title(f'{plot_title} :: Actual vs. Predicted Streams for\n{model} w/ PCA', fontsize=16)
			plt.grid(True, which='both', linestyle='--', linewidth=0.5)
			plt.tight_layout()
			plt.show()


	pd.DataFrame(bestConfigs).to_csv("results_csv/Singles_vs_colabs.csv")

	return bestConfigs


def menu():
	print("\n\t<< Menu >>")
	print('\n 1) naiveLoop()')
	print('\n 2) trySizesMLP()')
	print('\n 3) <no PCA> gridSearch for MLP and SVR')
	print('\n 4) <with PCA> gridSearch for MLP and SVR')
	print('\n 5) Best configurations')
	print('\n 6) Predictions for singles/colabs separately\n')

	option = int(input("Select one of the options above :: "))

	match option:
		case 1 :
			naiveLoop()
		case 2:
			trySizesMLP()
		case 3 :
			modelTuning(1)
		case 4 :
			# grid search with PCA for MLP
			df = gridSeach_PCA(1,True)
			df.to_csv("results_csv/gridSeach_PCA_MLP.csv")
			# grid search with PCA for SVR
			df = gridSeach_PCA(1,False)
			df.to_csv("results_csv/gridSeach_PCA_SVR.csv")
		case 5 :
			bestConfigs(1)
		case 6 :
			runSinglesvColabs()
		case _ :
			return

def main():

	menu()

if __name__ == "__main__":
	main()


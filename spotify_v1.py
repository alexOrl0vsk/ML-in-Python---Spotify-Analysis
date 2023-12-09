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


def loadData(raw):
	# input:  raw (bool) :: True -> original data, 
	#						False -> combine playlist/chart data 
	df = pd.read_csv('spotify2023.csv', encoding="ISO-8859-1")
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

def customPipe(caseID,scaler,model):
	try:
		X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(caseID)


		if {'key', 'mode'}.issubset(X_train.columns):
			encoder = make_column_transformer(
					(OneHotEncoder(), ['key','mode']),
					remainder='passthrough')
			myPipe = make_pipeline(encoder,scaler,model)
		else : 
			myPipe = make_pipeline(scaler,model)

		myPipe.fit(X_train,y_train)

		RMSE_train = mean_squared_error(y_train,myPipe.predict(X_train),squared=False)
		RMSE_valid = mean_squared_error(y_valid,myPipe.predict(X_valid),squared=False)


		return [RMSE_train,RMSE_valid] #{"RMSE_train": RMSE_train,"RMSE_valid":RMSE_valid}

	except ValueError:
		print("Select one of predefined cases!")
	return [np.nan, np.nan]

def featureCases(caseID):
	df = loadData()
	y = df['streams']
	match caseID:
		case 0:	# all features 
			X = df.drop(columns=['streams'])
			print(X.shape)
			encoder = make_column_transformer(
						(OneHotEncoder(), ['key','mode']),
						remainder='passthrough'
						)
			X = pd.DataFrame(encoder.fit_transform(X))
		case 1: # all features WIHOUT categorical variables 
			X = df.drop(columns=['streams','key','mode'])
		case 2: # custom selected feature cases 
			X = df[['artist_count','bpm']]
		case 3:
			X = df[['bpm','valence_%','speechiness_%']]
		case 4: # PCA without categorical, only first 5 components
			X = df.drop(columns=['streams','key','mode'])
			pca = PCA().set_output(transform="pandas").fit(X)
			X = pca.transform(X).iloc[:,:5]
		case 5: # PCA w/ categorical encoded , first 5 component 
			X = df.drop(columns=['streams'])
			encoder = make_column_transformer(
						(OneHotEncoder(), ['key','mode']),
						remainder='passthrough'
						)

			X_encoded = encoder.fit_transform(X)
			pca = PCA().set_output(transform="pandas").fit(X_encoded)
			X = pca.transform(X_encoded).iloc[:,:5]
		case _: 
			raise ValueError

	return getSplit(X,y)

def loopAll():
	models = [LinearRegression(),MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=10000, solver='adam')]  # ,svm.SVR()
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

	return

def bestConfig(caseID):
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(caseID)
	scaler = RobustScaler()
	modelMLP = MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=10000, solver='adam',alpha=0.0001)
	modelSVR = svm.SVR(kernel='rbf',
						degree = 4,
						C = 1000)

	models = [modelMLP, modelSVR]

	for model in models:
		myPipe = make_pipeline(scaler,model)
		myPipe.fit(X_train,y_train)

		RMSE_train = mean_squared_error(y_train,myPipe.predict(X_train),squared=False)
		RMSE_valid = mean_squared_error(y_valid,myPipe.predict(X_valid),squared=False)

		print(f'\n\n\tNo PCA, {model}')
		print(f'RMSE for training\t:: {RMSE_train}')
		print(f'RMSE for validation\t:: {RMSE_valid}')
		print(f'RMSE for test before validation fit\t:: {mean_squared_error(y_test,myPipe.predict(X_test),squared=False)}')
		myPipe.fit(pd.concat([X_train, X_valid], axis=0),pd.concat([y_train, y_valid], axis=0))
		print(f'RMSE for test after validation fit\t:: {mean_squared_error(y_test,myPipe.predict(X_test),squared=False)}')

		print(f"\n\t {model} w/ optimal PCA: ")
		pipePCA = make_pipeline(scaler,
								PCA(n_components=6).set_output(transform="pandas"),
								modelSVR
								)


		pipePCA.fit(X_train,y_train)
		print(f'RMSE for training\t:: {mean_squared_error(y_train,pipePCA.predict(X_train),squared=False)}')
		print(f'RMSE for validation\t:: {mean_squared_error(y_valid,pipePCA.predict(X_valid),squared=False)}')
		print(f'RMSE for test before validation fit\t:: {mean_squared_error(y_test,pipePCA.predict(X_test),squared=False)}')
		pipePCA.fit(pd.concat([X_train, X_valid], axis=0),pd.concat([y_train, y_valid], axis=0))
		print(f'RMSE for test after validation fit\t:: {mean_squared_error(y_test,pipePCA.predict(X_test),squared=False)}')
	return

def lab9():
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(1)
	scaler = RobustScaler()
	model = MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=20000, solver='adam')

	myPipe = make_pipeline(scaler,model)

	myPipe.fit(X_train,y_train)

	scores = cross_val_score(myPipe, X_train, y_train, cv=5,
							scoring='neg_root_mean_squared_error')


	for n in [3,5,10]:
		print(f'{n}-fold Cross-Validation :: ',cross_val_score(myPipe, X_train, y_train, cv=n,scoring='neg_root_mean_squared_error').mean())

	'''model_mse = list(np.multiply(model_nmse , -1))
				model_mae = list(np.multiply(model_nmae , -1))'''

	return

def gridSearch():
	warnings.filterwarnings("ignore")
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(1)

	scaler = RobustScaler()
	model = MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=20000, solver='adam')
	myPipe = make_pipeline(scaler,model)

	param_grid = {
	    'mlpregressor__hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
	    'mlpregressor__max_iter': [ 10000, 20000],
	    'mlpregressor__solver': ['adam'],
	    'mlpregressor__alpha': [0.0001, 0.0002, 0.0003],
	}

	grid = GridSearchCV(myPipe, param_grid,scoring=['neg_root_mean_squared_error','r2'],
						refit ='neg_root_mean_squared_error',
						n_jobs = -1,
						verbose = 3,
						cv = 5)

	grid.fit(X_train, y_train)

	print('\nBest params :: \n',grid.best_params_) 

	print('RMSE for train :: ',mean_squared_error(y_train,grid.predict(X_train),squared=False))
	print('RMSE for valid :: ',mean_squared_error(y_valid,grid.predict(X_valid),squared=False))
	print('RMSE for test :: ',mean_squared_error(y_test,grid.predict(X_test),squared=False))

	df_temp = pd.DataFrame(grid.cv_results_)
	df_temp = df_temp.sort_values("rank_test_r2")
	df_temp.to_csv()

def test():
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(1)
	scaler = RobustScaler()
	for state in np.arange(1,20):
		model = MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=state,alpha=0.0001,
							 activation='relu',max_iter=10000, solver='adam')

		myPipe = make_pipeline(scaler,model)

		myPipe.fit(X_train,y_train)
		print('\n \tState :: ',state)
		print(f'RMSE for training\t:: {mean_squared_error(y_train,myPipe.predict(X_train),squared=False)}')
		print(f'RMSE for validation\t:: {mean_squared_error(y_valid,myPipe.predict(X_valid),squared=False)}')
		print(f'RMSE for test before validation fit\t:: {mean_squared_error(y_test,myPipe.predict(X_test),squared=False)}')
		myPipe.fit(pd.concat([X_train, X_valid], axis=0),pd.concat([y_train, y_valid], axis=0))
		print(f'RMSE for test after validation fit\t:: {mean_squared_error(y_test,myPipe.predict(X_test),squared=False)}')


def PCA_pipe():
	# idea		:: 	general pipeline with different (scalers / no. of PCA components / models) with same step names 


	# output 	:: 
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(0)

	MyPipe = Pipeline([
		("scale", RobustScaler()),
		("PCA", PCA().set_output(transform="pandas")),
		("model", MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,alpha=0.0001,
							 activation='relu',max_iter=10000, solver='adam'))
		])

	param_grid = {
		'PCA__n_components' : np.arange(5,20),
	    'model__hidden_layer_sizes': [(15,15,10),(20,15,10,1)],
	    'model__max_iter': [ 10000, 20000],
	    'model__activation': ['relu','identity','logistic'],
	    'model__alpha': [0.0001, 0.0002, 0.0003]
	}

	grid = RandomizedSearchCV(MyPipe, param_grid, scoring=['neg_root_mean_squared_error','r2'],
						random_state = 1, 
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
	df_temp.to_csv("PCA2.csv")

	
	'''Best params ::
				 {'model__max_iter': 10000, 'model__hidden_layer_sizes': (15, 15, 10), 'model__alpha': 0.0001, 'model__activation': 'relu', 'PCA__n_components': 6}
				RMSE for train ::  192.36533638569816
				RMSE for valid ::  264.53603539014296
				RMSE for test ::  287.10365159610234'''


def trySizesMPL():
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(1)
	scaler = StandardScaler()
	model = MLPRegressor(hidden_layer_sizes=(15,15,10), random_state=1,max_iter=20000, solver='adam')
	myPipe = make_pipeline(scaler,model)
	
	param_grid = {
	    'mlpregressor__hidden_layer_sizes': [(20,20,15,10,1), (20,15,10,10,5), (20,15,10,5,1),(20,15,10,5,1), (20,15,10,1), (20,15,15,5),(20,15,15,1),(15,15,10,1), (15,10,10,1),
	    									(30,20,10,1),(20,15,5,1),(15,15,10),(20,20,15),(20,15,10),(20,10,5),(20,15,5)],
	    'mlpregressor__max_iter': [ 10000, 20000],
	    'mlpregressor__activation': ['tanh', 'relu','identity','logistic'],
	    'mlpregressor__solver': ['adam','lbfgs','sgd' ], #'lbfgs','sgd', 
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
	df_temp.to_csv("randd.csv")



def Models_tuning(featureCaseID):
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
						refit = 'neg_root_mean_squared_error',
						n_jobs = -1,
						verbose = 3,
						cv = 5,
						n_iter = 75
						)
		grid.fit(X_train,y_train)
		scores.append({
				'model' : model_name,
				'best_params' : grid.best_params_,
				'best_score' : grid.best_score_
			})


	return pd.DataFrame(scores)



def outliersDemo():
	X_train,X_valid,X_test,y_train,y_valid,y_test = featureCases(1)


	model = MLPRegressor(hidden_layer_sizes = (15,15,10),
							random_state = 1,
							max_iter = 10000,
							solver = 'adam',
							alpha = 0.0001)

	model.fit(X_train,y_train)
	print(f'MLP score :: {model.score(X_train,y_train)}')

	print(f'RMSE for training\t:: {mean_squared_error(y_train,model.predict(X_train),squared=False)}')
	print(f'RMSE for validation\t:: {mean_squared_error(y_valid,model.predict(X_valid),squared=False)}')

	pred_vs_real_train = pd.DataFrame({"Pred" : model.predict(X_train), "Real" : y_train})

	e_train = y_train - model.predict(X_train)
	print("Mean error (manual) :: ", e_train.mean())

	'''plt.hist(e_train,color='r')
				plt.grid(),plt.xlabel('Error [-]'), plt.title("Training Error Histogram")
				plt.show()
			
				plt.plot(e_train,'*')
				plt.grid(),plt.ylabel('Error [-]'), plt.title("Training Errors")
				plt.show()'''

	'''plt.figure(figsize=(10,6))
				sns.regplot(data=pred_vs_real_train, x="Pred" , y="Real")
				plt.grid(),
				plt.ylabel('Real stream count [millions]'),
				plt.xlabel('Predicted stream count [millions]'),
				plt.title("Training set real vs predicted target value")
			
				plt.show()'''


	from sklearn.covariance import EllipticEnvelope
	elp_env = EllipticEnvelope(contamination = 0.05) # 5 % removed
	pred = elp_env.fit_predict(np.array(pred_vs_real_train))

	print("Identified outliers' count :: ", list(pred).count(-1)) 
	print(X_train.shape)

	# outliers' indices 
	out_ids = np.where(pred == -1)
	out_ys = X_train.iloc[out_ids]
	print("out_ys :: ",out_ys)

	plt.figure(figsize=(10,6))
	plt.plot(X_train.iloc[:,5], X_train.iloc[:,6], '*', color = 'b',label = 'Normal samples')
	plt.plot(out_ys.iloc[:,5], out_ys.iloc[:,6], '*', color = 'r',label = 'Elliptic envelope outliers')
	plt.xlabel("feature 1")
	plt.ylabel("feature 2")
	plt.grid(),
	plt.show()


def main():
	#lab9()

	#df = Models_tuning(1)
	#df.to_csv('Models_tuning2.csv', mode='a', header=False)

	#bestConfig(1)

	#loopAll()
	#gridSVR()
	#trySizesMPL()
	#test()
	'''for id in [0 ,1 ]:
					print("\nFeature case ",id)
					bestConfig(id)'''
	#PCA_pipe()

	outliersDemo()

if __name__ == "__main__":
	main()
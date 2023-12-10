import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

# Custom functions' imports
from spotify_v1 import loadData

def getBasicVisuals():
	df = loadData(True)
	# Streams vs number of artists 
	fig = plt.figure(figsize=(5,5),layout="constrained")
	plt.scatter(df['artist_count'],df['streams'],marker='.')
	plt.title('Number of artists on a song vs Stream count')
	plt.xlabel('Number of artists')
	plt.ylabel('Stream count [in millions]')

	# box plots
	plt.figure(figsize=(20,10),layout="constrained")
	df.boxplot()

	# histograms
	fig = plt.figure(figsize = (15,10),layout="constrained")
	ax = fig.gca()
	df.hist(ax = ax)

	# key vs streams 
	plt.figure(figsize=(7,5))
	sns.scatterplot(x=df['key'],y=df['streams'],hue=df['mode'],marker='.')
	plt.title('Song Key vs Stream Count')
	plt.xlabel('Key [-]')
	plt.ylabel('Stream count [in millions]')
	plt.show()

	# stream correlation with playlists/charts
	df_corr= df[[ 'streams','in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists','in_deezer_charts','in_shazam_charts']].corr()
	plt.figure(figsize=(8,6))
	plt.title('Correlation between Stream count and presence in a playlist/chart')
	colormap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
	sns.heatmap(df_corr, annot= True, vmin=-1, vmax=1, center=0, cmap=colormap)
	plt.show()


	# Playlist/chart features are now combined 
	df = loadData(False)

	# Streams vs playlist presence
	plt.figure(figsize=(5,5))
	plt.scatter(df['playlist_presence'],df['streams'],marker='.')
	plt.title('Playlist Presence vs Stream Count')
	plt.xlabel('Number of Playlists a song is in [-]')
	plt.ylabel('Number of Streams a song receives [in millions]')

	# Correlation matrix
	if {'key', 'mode'}.issubset(df.columns):
		df = df.drop(columns=['key','mode'])
	corM= df.corr()
	plt.figure(figsize=(10,8))
	plt.title('Correlation Matrix ')
	colormap = sns.cubehelix_palette(start=.4, rot=-.5, as_cmap=True)
	sns.heatmap(corM, annot= True, vmin=-1, vmax=1, center=0, cmap=colormap)

	# combined playlist/char data to stream count correlation
	df_corr= df[[ 'streams','playlist_presence','chart_presence','artist_count']].corr()
	plt.figure(figsize=(8,6))
	plt.title('Correlation between Stream count and presence in a playlist/chart')
	colormap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
	sns.heatmap(df_corr, annot= True, vmin=-1, vmax=1, center=0, cmap=colormap)

	# song qualities correlation
	df_corr= df[['streams','playlist_presence','chart_presence','danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%','liveness_%','speechiness_%']].corr()
	plt.figure(figsize=(9,7))
	plt.title('Correlation between Stream count and Song qualities')
	colormap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
	sns.heatmap(df_corr, annot= True, vmin=-1, vmax=1, center=0, cmap=colormap)
	plt.show()


getBasicVisuals()
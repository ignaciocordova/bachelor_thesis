import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import do_functions as do 

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc



def read_and_merge(document1,document2):
	""" Reads documents 1 and 2 and merges 
		dataframes based on roid 
	
	Input: documents to read
	Output: df with all data merged
	"""
	metadata = pd.read_pickle(document1)
	profiles = pd.read_pickle(document2)
	df = pd.merge(metadata, profiles, on="roid")
	return df 

def clean_dataframe(df):
	df = df.drop(df[df.dphi_0010 == -999.0].index)
	df = df.drop(df[df.meanP_2 < 0.0].index)
	df = df.drop(df[df.precipBelow12 < 0.0].index)

<<<<<<< HEAD
	#Make the values of hxxx below height_flag_comb Nan
	#values = df.loc[:,'h001':'h400'].to_numpy()
	#flag = df.loc[:,'height_flag_comb'].to_numpy()
	#for el in flag:
		#index = int(el)
		#values[:,0:index] = np.nan
	
	#df.loc[:,'h001':'h400'] = pd.DataFrame(values)

=======
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4
	return df 

def plot_profile(df,roid):
	""" Plots the vertical profile of measure 'roid'
	
	Input: roid
	Output: just displays a plot 
	"""
	df.loc[df['roid']==roid].iloc[0,10:].plot()
<<<<<<< HEAD


=======
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4
	plt.xlabel('Height')
	plt.ylabel('Δɸ')
	plt.show()
    

def average_dphi(dfin,hi,hf,dfout):
	""" Averages dphi from dataframe(in) between hi and hf
		and writes the result for each measurement in df(out)

	Input: dataframe with dphi data, hi,hf, df to save averages
	Output: returns 'result' as dfout with each new average 
			it also modifies dfout by adding the calculated average  
	"""

	# if hf<hi tell me and don't average 
	if hf<=hi:
	    print('hf must be above hi and '+str(hf)+' is not above '+str(hi))
	    return 
	 
	#convert float inputs to string to find labels in dataframe 
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)
	    
	#build auxiliary dataframe2 to calculate mean 
	indx1 = dfin.columns.get_loc(str1)
	indx2= dfin.columns.get_loc(str2)

	df2= dfin.iloc[:,indx1:indx2+1] #only the columns to average

	########Data cleaning########
	#If NaN values are more than 20%, converts the whole row to NaN
	df2.loc[df2.isna().sum(axis=1) > len(df2.columns)/5, :] = np.nan

	# writes the calculated average for each measure
	result = dfout
	result['avg'+str1+''+str2] = df2.mean(axis=1)

	return result


def average(dfin,hi,hf):
	""" Averages dphi from dataframe(in) between hi and hf
		and outputs a 2 column dataframe with average for 
		each measure and column of precipBelow12

	Input: dataframe with dphi data, hi,hf
	Output: returns 'result' with each new average  and 
			column of precipBelow12
	"""

	# if hf<hi tell me and don't average 
	if hf<=hi:
	    return 
	
	#convert float inputs to string to find labels in dataframe 
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)
	    
	#build auxiliary dataframe2 to calculate mean 
	indx1 = dfin.columns.get_loc(str1)
	indx2= dfin.columns.get_loc(str2)

	df2= dfin.iloc[:,indx1:indx2+1] #only the columns to average

	########Data cleaning########
	#If NaN values are more than 20%, converts the whole row to NaN
	df2.loc[df2.isna().sum(axis=1) > len(df2.columns)/5, :] = np.nan
<<<<<<< HEAD
	
=======

>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4
	# writes the calculated average for each measure
	result = pd.DataFrame()
	result['avg'+str1+''+str2] = df2.mean(axis=1)
	result['precipBelow12'] = dfin['precipBelow12'].copy()
<<<<<<< HEAD
	result[result['avg'+str1+''+str2] < 0.0] = 0.
	
=======
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

	#drops averages that have outputted NaN 
	#this way precipBelow12 and avg have the same length 
	result.dropna(inplace=True)

	return result


<<<<<<< HEAD
def plotROC(df,hi,hf,percentile):
=======
def plotROC(df,hi,hf,truth_th):
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

	df_avg = do.average(df,hi,hf)
	# 2 columns: precipBelow12 and avg have the same length

	# if hf<hi don't bother
	if hf<=hi:
		return

	#convert float inputs to string to find labels in dataframe
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)

	#build numpy array with normalized averages (there are negative values!)
	probs = (df_avg['avg'+str1+''+str2]/df_avg['avg'+str1+''+str2].max()).to_numpy()

<<<<<<< HEAD
	#Build boolean Truth array with True above percentile 
	#the percentile ignores 0 values  
	auxiliary_df = df_avg[df_avg['precipBelow12']>0]
	truth_th = auxiliary_df['precipBelow12'].quantile(percentile)
	truth = (df_avg['precipBelow12']>truth_th).to_numpy()
=======
	#Build boolean Truth array using truth threshold 
	truth = (df_avg['precipBelow12']/df_avg['precipBelow12'].max()>truth_th).to_numpy()
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

	#Calculate ROC curves fpr = false positive rate ; tpr = true positives rate 
	fpr, tpr, _ = roc_curve(truth, probs)

	#Generate ROC curve for random no-skill model
	ns_probs = [0 for _ in range(len(truth))]
	ns_fpr, ns_tpr, _ = roc_curve(truth, ns_probs)

	#and plot them 
	plt.plot(ns_fpr, ns_tpr, linestyle='--',c='gray')
<<<<<<< HEAD
	plt.plot(fpr, tpr, marker='.', label=str(hi)+'km -'+str(hf)+'km'+'; percentile='+str(round(percentile*100,3)))
	plt.legend(bbox_to_anchor =(1.0, 1.0))
	return

def plotPrecisionRecall(df,hi,hf,percentile):
=======
	plt.plot(fpr, tpr, marker='.', label=str(hi)+'km -'+str(hf)+'km')
	plt.title('ROC curve for precipitation threshold of '+str(round(truth_th*df_avg['precipBelow12'].max(), 2)))
	plt.legend(bbox_to_anchor =(1.0, 1.0))
	return

def plotPrecisionRecall(df,hi,hf,truth_th):
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

	df_avg = do.average(df,hi,hf)
	# 2 columns: precipBelow12 and avg have the same length

	# if hf<hi don't bother
	if hf<=hi:
		return

	#convert float inputs to string to find labels in dataframe
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)

	#build numpy array with normalized averages (there are negative values!)
	probs = (df_avg['avg'+str1+''+str2]/df_avg['avg'+str1+''+str2].max()).to_numpy()

<<<<<<< HEAD
	#Build boolean Truth array with True above percentile 
	#the percentile ignores 0 values  
	auxiliary_df = df_avg[df_avg['precipBelow12']>0]
	truth_th = auxiliary_df['precipBelow12'].quantile(percentile)
	truth = (df_avg['precipBelow12']>truth_th).to_numpy()
=======
	#Build boolean Truth array with threshold 
	truth = (df_avg['precipBelow12']/df_avg['precipBelow12'].max()>truth_th).to_numpy()
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

	# calculate precision and recall curve
	precision, recall, thresholds = precision_recall_curve(truth, probs)

	no_skill = len(truth[truth==1]) / len(truth)
	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', c='black')


	#the optima threshold based on greater F1 SCORE 
	f1_scores = (2.0*precision*recall)/(precision+recall)
<<<<<<< HEAD
	i = np.nanargmax(f1_scores)
=======
	i = np.argmax(f1_scores)
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

	if np.isnan(f1_scores[i])==True:
		print('The avg_dphi',str1,'-',str2,' has NO SKILL')
		print('')


	else: 
		#averaged dphi is normalized so we must multiply by avg.max()
		print('For the avg_dphi',str1,'-',str2,':')
		print('Optimal threshold: =',round(thresholds[i]*df_avg['avg'+str1+''+str2].max(),2))
		print('F1score:',round(f1_scores[i],3))
		print('')


	#and plot them
<<<<<<< HEAD
	plt.plot(precision, recall, marker='.', label=str(hi)+'km -'+str(hf)+'km'+'; percentile='+str(round(percentile*100,3)))
	plt.plot(precision[i], recall[i], color='red', marker='o',linewidth=2, markersize=7)
=======
	plt.plot(precision, recall, marker='.', label=str(hi)+'km -'+str(hf)+'km')
	plt.plot(precision[i], recall[i], color='red', marker='o',linewidth=2, markersize=7)
	plt.title('Precision-Recall curve for precipitation threshold of '+str(round(truth_th*df_avg['precipBelow12'].max(), 2)))
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4
	plt.legend(bbox_to_anchor =(1.0, 1.0))

	return 


<<<<<<< HEAD



def getF1score(df,hi,hf,percentile):
	#the function returns a list with the results 
=======
def getF1score(df,hi,hf,truth_th):
	#the function returns optimal th and F1 score of optimal th
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

	df_avg = do.average(df,hi,hf)
	# 2 columns: precipBelow12 and avg have the same length

	# if hf<hi don't bother
	if hf<=hi:
		return

	#convert float inputs to string to find labels in dataframe
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)

	#build numpy array with normalized averages (there are negative values!)
	probs = (df_avg['avg'+str1+''+str2]/df_avg['avg'+str1+''+str2].max()).to_numpy()

<<<<<<< HEAD
	#Build boolean Truth array with True above percentile 
	#the percentile ignores 0 values  
	auxiliary_df = df_avg[df_avg['precipBelow12']>0]
	truth_th = auxiliary_df['precipBelow12'].quantile(percentile)
	truth = (df_avg['precipBelow12']>truth_th).to_numpy()
=======
	#Build boolean Truth array with threshold 
	truth = (df_avg['precipBelow12']/df_avg['precipBelow12'].max()>truth_th).to_numpy()
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

	# calculate precision and recall curve
	precision, recall, thresholds = precision_recall_curve(truth, probs)

	#the optima threshold based on greater F1 SCORE 
<<<<<<< HEAD
	f1_scores = (2.0*precision*recall)/(precision+recall)
	i = np.nanargmax(f1_scores)
	

	#the function returns a list with the results 
	return [hi,hf,truth_th,thresholds[i]*df_avg['avg'+str1+''+str2].max(),f1_scores[i]]


=======
	f1 = f1_score(truth, probs)

	#the function returns optimal th and F1 score of optimal th
	return hi,hf,truth_th,thresholds[i]*df_avg['avg'+str1+''+str2].max(),f1
>>>>>>> 5b72414f9dab1f787a6c68efb2e642a2d1419cf4

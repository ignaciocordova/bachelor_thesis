import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import paz
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


"""
############################################################

The PAZ Python module was developed by Ignacio Cordova Pou during
2022 for his final thesis at the Universitat de Barcelona.
The module is used for the paper: "Performance Evaluation of
Polarimetric Radio Occultations Measurements in Detecting 
Precipitation" and contains funtions to work with the data 
gathered in the ROHP-PAZ mission. To acces the data please
contact ICE-CSIC at https://www.ice.csic.es/ . 

###############################################################
 """


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
	""" 
	Cleans the dataframe by removing rows with problematic 
	values like negative precipitation or negative heights. 

	Input: dataframe
	Output: dataframe with cleaned data
	"""
	#removes rows with problematic values 
	df = df.drop(df[df.dphi_0010 == -999.0].index)
	df = df.drop(df[df.meanP_2 < 0.0].index)
	df = df.drop(df[df.precipBelow6 < 0.0].index)
	df = df.drop(df[df.height_flag_comb < 0.0].index)

	#Converts the values of hxxx below height_flag_comb to Nan
	values = df.loc[:,'h001':'h400'].to_numpy()
	flag = df.loc[:,'height_flag_comb'].to_numpy()
	col = df.loc[:,'h001':'h400'].columns

	i=0
	for el in flag:
		index = int(10.0*el)
		#all the values below the flag are converted to Nan
		values[i,:index] = np.nan
		i = i+1 
	
	#substitues the values in the original dataframe
	df.loc[:,'h001':'h400'] = pd.DataFrame(values,columns = col).to_numpy()
	
	return df

def separate_by_region(df):
	""" Separates the dataframe into two dataframes based on region
		and returns the two dataframes
		
		Input: dataframe
		Output: dataframes with data separated by region LABEL
		"""
	land_df = df[df['region']==0]
	sea_df = df[df['region']!=0]

	return land_df,sea_df

def separate_tropics(df):
	""" Separates the dataframe into two dataframes based latitude 
	and longitude values and region labels corresponding to tropical 
	regions and extratropical regions
	
	Input: dataframe
	Output: dataframes with data separated by tropics and extratropics
	"""
	extratropics_df = df[(df['region']==1) | ((df['region']==0)&(abs(df['lat'])>30)) ]
	tropics_df = df.drop(df[(df['region']==1) | ((df['region']==0)&((df['lat']>30)|(df['lat']<-30) )) ].index)
	return tropics_df,extratropics_df
		
	

def plot_profile(df,roid):
	""" Plots the vertical profile of a PRO. The measure is identified by 
	the column roid.
	
	Input: dataframe, roid
	Output: none, displays a plot 
	"""
	df.loc[df['roid']==roid].iloc[0,12:].plot()


	plt.xlabel('Height')
	plt.ylabel('Δɸ')
	plt.show()

def get_profile(df,i):
	""" Returns the vertical profile of a PRO. The measure is identified by 
	the index. It returns a numpy array with the vertical profile. 
	
	Input: datafram and index i 
	Output: numpy array with Δɸ for each height
	"""

	v_profile = df.iloc[i,12:].to_numpy()

	return v_profile

def get_profile_from_roid(df,roid):
	""" Returns the vertical profile of a PRO. The measure is identified by 
	the ROID. It returns a numpy array with the vertical profile. 
	
	Input: datafram and roid 
	Output: numpy array with Δɸ for each height
	"""

	v_profile = df.loc[df['roid']==roid].iloc[0,12:].to_numpy()

	return v_profile

    

def average_dphi(dfin,hi,hf,dfout):
	""" Averages dphi from dataframe(dfin) between hi and hf
		and writes the result for each measurement.
		Note that dfout is not modified but a new dataframe is created
		with the same data plus the averages computed. 
		

	Input: dataframe with dphi data, hi,hf, and dfout to save averages
	Output: returns a dataframe like dfout but with each new average. 
	"""

	# if hf<hi tell me and don't average 
	if hf<=hi:
	    print('hf must be above hi and '+str(hf)+' is not above '+str(hi))
	    return 
	 
	#convert float inputs to string to find labels in dataframe 
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)
	    
	#build auxiliary dataframe2 to calculate the average 
	indx1 = dfin.columns.get_loc(str1)
	indx2= dfin.columns.get_loc(str2)

	df2= dfin.iloc[:,indx1:indx2+1] #only the columns to average

	########Important condition########
	#If NaN values are more than 33%, converts the whole row to NaN
	df2.loc[df2.isna().sum(axis=1) > len(df2.columns)/3.0, :] = np.nan

	# writes the calculated average for each measure 
	# Note that dfout is not modified but a new dataframe is created
	# with the same data plus the averages computed.
	result = dfout
	result['avg'+str1+''+str2] = df2.mean(axis=1)

	return result


def average(dfin,hi,hf):
	""" Averages dphi from dataframe(in) between hi and hf
		and outputs a 2-column dataframe with the average for 
		each measure and a column of precipBelow6 (true precipitation)

	Input: dataframe with dphi data, hi,hf
	Output: returns dataframe 'result' with each new average  and a 
			column of precipBelow6
	"""

	# if hf<hi don't average 
	if hf<=hi:
	    return 
	
	#convert float inputs to string to find labels in dataframe 
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)
	    
	#build auxiliary dataframe2 to calculate average 
	indx1 = dfin.columns.get_loc(str1)
	indx2= dfin.columns.get_loc(str2)

	df2= dfin.iloc[:,indx1:indx2+1] #only the columns to average

	########Important condition########
	#If NaN values are more than 33%, converts the whole row to NaN
	df2.loc[df2.isna().sum(axis=1) > len(df2.columns)/3.0, :] = np.nan
	
	# the results go into a new df
	result = pd.DataFrame()
	# writes the calculated average for each PRO
	result['avg'+str1+''+str2] = df2.mean(axis=1)
	# copies the true precipitation column
	result['precipBelow6'] = dfin['precipBelow6'].copy()
	# some close to zero negative values are converted to zero
	result[result['avg'+str1+''+str2] < 0.0] = 0.
	

	#drops averages that have outputted NaN 
	#this way precipBelow6 and avg have the same length 
	result.dropna(inplace=True)

	return result


def plotROC(df,hi,hf,percentile):
	""" Plots the ROC curve of an average for a given percentile.

	Input: dataframe with metadata, hi,hf, and percentile
	Output: none, displays a plot
	"""

	df_avg = paz.average(df,hi,hf)
	# paz.average returns datafram with 2 columns: precipBelow6 
	# and avg, they have the same length. 

	# if hf<hi don't bother
	if hf<=hi:
		return

	#convert float inputs to string to find labels in dataframe
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)

	#build numpy array with NORMALIZED averages (they will be treated as probabilities)
	probs = (df_avg['avg'+str1+''+str2]/df_avg['avg'+str1+''+str2].max()).to_numpy()

	#Build boolean Truth array with True for precipitation above percentile  
	auxiliary_df = df_avg[df_avg['precipBelow6']>0] #the percentile ignores 0 values 
	truth_th = auxiliary_df['precipBelow6'].quantile(percentile) #sets the truth threshold
	truth = (df_avg['precipBelow6']>truth_th).to_numpy() #binary target numpy array

	#ROC curves: fpr = false positive rate ; tpr = true positives rate 
	fpr, tpr, _ = roc_curve(truth, probs)

	#Generate ROC curve for random no-skill model
	ns_probs = [0 for _ in range(len(truth))] #random probabilities 
	ns_fpr, ns_tpr, _ = roc_curve(truth, ns_probs)

	#and plot them 
	plt.plot(ns_fpr, ns_tpr, linestyle='--',c='gray')
	plt.plot(fpr, tpr, marker='.', label=str(hi)+'km -'+str(hf)+'km'+'; percentile='+str(round(percentile*100,3)))
	plt.legend(bbox_to_anchor =(1.0, 1.0))
	return

def plotPrecisionRecall(df,hi,hf,percentile):
	""" Plots the Precision-Recall curve of an average for a given percentile.

	Input: dataframe with metadata, hi,hf, and percentile
	Output: none, displays a plot
	"""

	df_avg = paz.average(df,hi,hf)
	# paz.average returns datafram with 2 columns: precipBelow6 
	# and avg, they have the same length.

	# if hf<hi don't bother
	if hf<=hi:
		return

	#convert float inputs to string to find labels in dataframe
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)

	#Eliminates measures with high dphi and low precipitation
	#this could be used to check if performance improves 
	#df_avg = df_avg.drop(df_avg[(df_avg['avg'+str1+''+str2] >0.7) & (df_avg['precipBelow6']<0.7)].index)

	#build numpy array with NORMALIZED averages (they will be treated as probabilities)
	probs = (df_avg['avg'+str1+''+str2]/df_avg['avg'+str1+''+str2].max()).to_numpy()

	#Build boolean Truth array with True for precipitation above percentile   
	auxiliary_df = df_avg[df_avg['precipBelow6']>0]#the percentile ignores 0 values
	truth_th = auxiliary_df['precipBelow6'].quantile(percentile)#sets the truth threshold
	truth = (df_avg['precipBelow6']>truth_th).to_numpy() #binary target numpy array

	# calculate precision-recall curve
	precision, recall, thresholds = precision_recall_curve(truth, probs)

	#generate precision-recall curve for random no-skill model
	no_skill = len(truth[truth==1]) / len(truth)
	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', c='black')

	#the optima threshold based on the largest F1 SCORE 
	f1_scores = (2.0*precision*recall)/(precision+recall)
	i = np.nanargmax(f1_scores)

	if np.isnan(f1_scores[i])==True:
		print('The avg_dphi',str1,'-',str2,' has NO SKILL')
		print('')

	else: 
		#remember averaged dphi is normalized so we must multiply by avg.max()
		print('For the avg_dphi',str1,'-',str2,':')
		print('Optimal threshold: =',round(thresholds[i]*df_avg['avg'+str1+''+str2].max(),2))
		print('Best F1score:',round(f1_scores[i],3))
		print('')

	#and plot them
	plt.scatter(precision, recall, 
				label=str(hi)+'km -'+str(hf)+'km'+'; percentile='+str(round(percentile*100,3)),
				s=1)
	plt.plot(precision[i], recall[i], color='red', marker='o', markersize=7)
	plt.legend(bbox_to_anchor =(1.0, 1.0))

	return 



def getF1score(df,hi,hf,percentile):
	""" Calculates the F1 score of an average for a given percentile.
	
	Input: dataframe with metadata, hi,hf, and percentile
	Output: list with [hi, hf, truth_th, avg_dphi th, best F1Score]
	"""

	df_avg = paz.average(df,hi,hf)
	# paz.average returns datafram with 2 columns: precipBelow6 
	# and avg, they have the same length.

	# if hf<hi don't bother
	if hf<=hi:
		return

	#convert float inputs to string to find labels in dataframe
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)

	#Eliminates measures with high dphi and low precipitation
	#this could be used to check if performance improves 
	#df_avg = df_avg.drop(df_avg[(df_avg['avg'+str1+''+str2] >0.7) & (df_avg['precipBelow6']<0.7)].index)


	#build numpy array with NORMALIZED averages (they will be treated as probabilities)
	probs = (df_avg['avg'+str1+''+str2]/df_avg['avg'+str1+''+str2].max()).to_numpy()

	#Build boolean Truth array with True for precipitation above percentile   
	auxiliary_df = df_avg[df_avg['precipBelow6']>0] #ignores zero values
	truth_th = auxiliary_df['precipBelow6'].quantile(percentile) #sets the truth threshold
	truth = (df_avg['precipBelow6']>truth_th).to_numpy() #binary target numpy array

	# calculate precision and recall curve
	precision, recall, thresholds = precision_recall_curve(truth, probs)

	#the optimal threshold based on the largest F1 SCORE 
	f1_scores = (2.0*precision*recall)/(precision+recall)
	i = np.nanargmax(f1_scores)
	

	#the function returns a list with the results 
	return [hi,hf,truth_th,thresholds[i]*df_avg['avg'+str1+''+str2].max(),f1_scores[i]]


def getPrecisionRecall(df,hi,hf,percentile):
	""" Calculates the Precision and Recall of an average for a given percentile.

	Input: dataframe with metadata, hi,hf, and percentile
	Output: precision array, recall array, best precision,
	best recall, array of truths and a label (string) to identify the result
	"""

	df_avg = paz.average(df,hi,hf)
	# paz.average returns datafram with 2 columns: precipBelow6 
	# and avg, they have the same length.

	# if hf<hi don't bother
	if hf<=hi:
		return

	#convert float inputs to string to find labels in dataframe
	str1 = 'h'+str(int(hi*10)).zfill(3)
	str2 = 'h'+str(int(hf*10)).zfill(3)

	#Eliminates measures with high dphi and low precipitation
	#this could be used to check if performance improves
	#df_avg = df_avg.drop(df_avg[(df_avg['avg'+str1+''+str2] >0.7) & (df_avg['precipBelow6']<0.7)].index)


	#build numpy array with NORMALIZED averages (they will be treated as probabilities)
	probs = (df_avg['avg'+str1+''+str2]/df_avg['avg'+str1+''+str2].max()).to_numpy()

	#Build boolean Truth array with True for precipitation above percentile
	auxiliary_df = df_avg[df_avg['precipBelow6']>0] #ignores zero values
	truth_th = auxiliary_df['precipBelow6'].quantile(percentile) #sets the truth threshold
	truth = (df_avg['precipBelow6']>truth_th).to_numpy() #binary target numpy array

	# calculate precision and recall curve; and the thresholds
	precision, recall, thresholds = precision_recall_curve(truth, probs)


	#the optima threshold based on the largest F1 SCORE 
	f1_scores = (2.0*precision*recall)/(precision+recall)
	i = np.nanargmax(f1_scores)

	if np.isnan(f1_scores[i])==True:
		print('The avg_dphi',str1,'-',str2,' has NO SKILL')
		print('')


	else: 
		#averaged dphi is normalized so we must multiply by avg.max()
		print('For the avg_dphi',str1,'-',str2,':')
		print('Optimal threshold: =',round(thresholds[i]*df_avg['avg'+str1+''+str2].max(),2))
		print('F1score:',round(f1_scores[i],3))
		print('')


	label=str(hi)+'km -'+str(hf)+'km'+'; percentile='+str(round(percentile*100,3))

	return precision,recall,precision[i], recall[i],truth,label


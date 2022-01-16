# This is the code for the confusion matrix 
## The following is the code for MLIS Course work 2021 - 2022
## Author 
## Anshuman Singh 
## Group members - Anshuman Singh, Alpaslan Erdag, Yixin Fan
## Group number - 1A 

import numpy as np

def confusion_matrix(y_true,y_pred):
	'''This function will returnn the coffusion matrix with false positives rate and recal rate'''
	false_positives = 0
	false_negatives = 0

	true_positives  = 0
	true_negatives  = 0

	for true_value, predicted_value in zip(y_true, y_pred):
    if predicted_value == true_value: #if the sample is officially true
        if predicted_value == 1: # true positves
            true_positives  += 1
        else: # true negaives
            true_negatives  += 1
    else: # false values?
        if predicted_value == 1: # false positives
            false_positives += 1
        else: # false negatives
            false_negatives += 1
            
	cm = [[true_negatives , false positives],
	  	  [false_negatives, true_positives]]
	# Converting the 2D list to 2D numpy array 
	cm = np.array(cm)
	
	false_positive_rate  = (false_positives) / (true_negatives+false_positives)
	recall = (true_positives) / (true_positives+false_negatives)

	return cm,false_positive_rate,recall

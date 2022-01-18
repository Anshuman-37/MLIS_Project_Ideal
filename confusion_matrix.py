# This is the code for the confusion matrix 
## The following is the code for MLIS Course work 2021 - 2022
## Author 
## Anshuman Singh 
## Group members - Anshuman Singh, Alpaslan Erdag, Yixin Fan
## Group number - 4B


# Header File
import numpy as np

# Code Reference 
# https://www.nbshare.io/notebook/626706996/Learn-And-Code-Confusion-Matrix-With-Python/

def confusion_matrix(y_true,y_pred):
    
    
    '''This function will returnn the coffusion matrix with false positives rate and recal rate'''
    false_positives = 0
    false_negatives = 0
    true_positives  = 0
    true_negatives  = 0

    # for true_value, predicted_value in zip(y_true, y_pred):
    #     if predicted_value == true_value:#if the sample matches 
    #         if predicted_value == 1:# true positves
    #             true_positives  += 1
    #         else: # true negaives
    #             true_negatives  += 1
    #     else: # Values don't match 
    #         if predicted_value == 1:# false positives
    #             false_positives += 1
    #         else:
    #             # false negatives   
    #             false_negatives += 1
    for true_value, predicted_value in zip(y_true, y_pred):
        if true_value == 1 and predicted_value == 1:
            true_positives+=1
        elif true_value == 0 and predicted_value == 1:
            false_positives+=1
        elif true_value == 0 and predicted_value == 0:
            true_negatives+=1
        elif true_value == 1 and predicted_value == 0:
            false_negatives+=1
            
    cm = [[true_positives , false_positives], [false_negatives, true_negatives]]
    
    # Converting the 2D list to 2D numpy array 
    cm = np.array(cm)
    
    # Fpr rate
    false_positive_rate  = (false_positives) / (true_negatives+false_positives)
    
    # Recall Rate
    recall = (true_positives) / (true_positives+false_negatives)
    
    #Auc score
    AUC_score = (true_positives+true_negatives)/(true_positives+true_negatives\
        +false_positives+false_negatives)
    
    # Precision 
    precision = (true_positives)/(true_positives + false_positives)

    # F1 score 
    F1_score  = (2*precision*recall)/(precision+recall)

    # Returning all the values 
    return cm,false_positive_rate,recall,AUC_score,precision,F1_score

""" 

# Author: Joey M Nelson
# 2/20/2016
# Runtime Environment: Python 3.5

Machine Learning >> Binary Classification >> Naive Bayes

"""

import sys
import csv
import random
import math
def Read_CSV(fileName):
    ''' Rawdata[0] = IV labels
        Rawdata[1:rowCt] = data values
        Rawdata[::][end] = DV classification label
    '''
    head = 0
    Rawdata = [[]] # 2 dimensional list of each data record
    rowCt = 0
    with open(fileName, newline='') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            a_list = []
            for i in row:
                a_list.append(i)
            if (head == 1):
                Rawdata.append( list(map(int, a_list)) )  # avoids appending to empty array and instead fils in first cell
                #print(a_list)
            else:
                Rawdata[0] = a_list
                head = 1
            rowCt += 1
    Labels = Rawdata.pop(0)        
    return Rawdata, Labels

if __name__ == "__main__":
    
    if (len(sys.argv) != 5):
        raise Exception("Incorrect number of arguments. correct usage: ./perceptron <train> <test> <beta> <model>")
        
    train = sys.argv[1]
    test = sys.argv[2]
    beta = float(sys.argv[3])
    Model = sys.argv[4]
    
    
    ## Train
    data_train, label_train = Read_CSV(train)
    data_train_len = len(data_train)
    label_train_len = len(label_train)
    
    num_straights = [0]*label_train_len
    P_straight = [0]*label_train_len
    
    num_X_givenC = [0]*label_train_len
    P_X_givenC = [0]*label_train_len
    
    P_C_givenX =[0] * label_train_len
    
    
    for a in range( data_train_len ): 
        for b in range( label_train_len):
            num_straights[b] = num_straights[b] + data_train[a][b]
            
            #Count num_X_givenC
            if ( (data_train[a][-1] == 1) and (data_train[a][b] == 1) ):
                num_X_givenC[b] = num_X_givenC[b] + 1
            
    # Convert counts to probabilities
    for i in range( label_train_len ):
        num_straights[i] += (beta -1) # Add beta to counts
        P_straight[i] = ( num_straights[i]) / ( data_train_len + (label_train_len*beta - label_train_len)  )
        P_X_givenC[i] = num_X_givenC[i]/num_straights[-1]
        
    # Calculate probabilities    
    for i in range( label_train_len ):  
        P_C_givenX[i] = (P_X_givenC[i] * P_straight[-1]) / (P_straight[i])
        
        
    ## Test
    data_test, label_test = Read_CSV(test)
    data_test_len = len(data_test)
    label_test_len = len(label_test)
    
    results = [0]*data_test_len
    correct = 0
    
    for i in range(data_test_len):
        P_calc = 1
        for atr in range (label_test_len):
            if (data_test[i][atr] == 1):
                P_calc = P_calc*P_C_givenX[atr]
                print(P_calc)
        if ( (P_calc > 0.50 and data_test[i][-1] == 1) or (P_calc < 0.50 and data_test[i][-1] == 0) ):
            results[i] = 1
        else:
            results[i] = 0
        correct += results[i]
        
    accuracy = correct/data_test_len
    #print("accuracy =", accuracy)
    
    ## Write Out Model
    f = open(Model, "w")
    logOdds = math.log2(num_straights[-1]) / math.log2( (data_train_len + (label_train_len*beta - label_train_len)) )
    f.write( str(logOdds) )
    f.close()
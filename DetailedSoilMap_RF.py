# -*- coding: utf-8 -*-

import csv
import time
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics

# =============================================================================
# functions
# =============================================================================

def ReadFile():
    
    global soil_X, soil_y, features, target_feature, target_feature_values
    
    print
    print 'reading csv file ...'
    print
    
    with open('DetailedSoilMap.csv','r') as csvfile:
        
        # read the file
        file_reader = csv.reader(csvfile)
        
        # go to the first line (columns)
        row = next(file_reader)
        
        #features = np.array(row)
        #print features
             
        # read all the data
        for row in file_reader:
            
            # delete y=0
            if ( row[6]!='0' ):  # delete y=1 by adding &( row[6]!='1' )
                soil_X_one_row = []
                
                for i in range(4,15,1):
                #for i in range(4,8,1):
                    
                    if i==6 :
                        soil_y.append( int(row[i]) )
                    else:
                        soil_X_one_row.append( int(row[i]) )
                    
                soil_X.append( soil_X_one_row )
    
    
    #print str(data[1]).decode('string_escape') 
            
    print 'target feature: ' , str(target_feature).decode('string_escape') 
    target_feature_values = list( set(soil_y) )
    print 'target feature values: ', target_feature_values
    
    print 'input features: ', len(features), ' features'
    print str(features).decode('string_escape') 
    
    print 'first line of data (input) : ', soil_X[0]
    print 'first line of data (output): ', soil_y[0]
    
    print 'second line of data (input) : ', soil_X[1]
    print 'second line of data (output): ', soil_y[1]
    
    print 'length of the data: ', len(soil_y)
    print 'number of y=1: ',soil_y.count(1)
    
      
    print      
    time2 = time.time()
    print 'current time =', time2-time1 , 'sec'
    print
    
    return


def HotEncoding():
    
    global soil_X, features
    
    print
    print 'hot encoding ...'
    print
    
    # add catagory numbers after the features
    features_he = []
    
    for i in range( len(features) ):
        
        one_feature_values = list( set([row[i] for row in soil_X]) )
        
        for j in range( len(one_feature_values) ):
            features_he.append( features[i]+str(one_feature_values[j]) )
            
    print 'input features: ', len(features_he), 'features'
    print str(features_he).decode('string_escape')       
    
    # spread input data into boolean form
    soil_X_he = soil_X
    
    for i in range( len(features) ):
        onehotencoder = OneHotEncoder(categorical_features = [len(soil_X_he[0])-1])
        #onehotencoder.fit(soil_X)
        #print onehotencoder.categories_
    
        soil_X_he = onehotencoder.fit_transform(soil_X_he).toarray()
    
        #print onehotencoder.get_params
    
    print 'first line of data (input): ', soil_X_he[0]
    print 'length: ', len(soil_X_he[0])
    print 'second line of data (input): ', soil_X_he[1]

    print 'third line of data (input): ', soil_X_he[2]
    
    # overwrite data
    soil_X = soil_X_he
    features = features_he

    print      
    time2 = time.time()
    print 'current time =', time2-time1 , 'sec'
    print    

    return


def SplitData(TEST_SIZE):
    
    global X_train, X_test, y_train, y_test
    
    print
    print 'splitting data ...'
    print
        
    X_train, X_test, y_train, y_test = train_test_split( soil_X, soil_y, 
                                                         test_size=TEST_SIZE, random_state=42 )
    
    print 'training sample size: ', len(X_train)
    print 'testing sample size:  ', len(X_test)
    
    print      
    time2 = time.time()
    print 'current time =', time2-time1 , 'sec'
    print
    
    return


def DT( MAX_DEPTH, CRITERION ):
    
    global X_train, X_test, y_train, y_test, features, target_feature_values
    
    # built a DT classifier
    tree_classifier = DecisionTreeClassifier( max_depth=MAX_DEPTH,
                                              random_state=1,
                                              criterion=CRITERION )
    # apply to training data
    tree_classifier = tree_classifier.fit(X_train,y_train)
    
    # predict testing data
    print 'first ten prediction:   ', tree_classifier.predict( X_test )[:10]
    print 'first ten actual value: ', y_test[:10]
    
    # print performance
    report = metrics.precision_recall_fscore_support( y_test,tree_classifier.predict( X_test ),
                                                      average='macro' )
    print 'accuracy:  ', tree_classifier.score( X_test, y_test )
    print 'precision: ', report[0]
    print 'recall:    ', report[1]
    print 'fscore:    ', report[2]
    
    # visualize tree
    export_graphviz( tree_classifier, out_file='tree.dot',
                     feature_names=features,
                     class_names = [format(x, 'd') for x in target_feature_values],
                     rounded = True, filled = True )
    
    # command line: dot -Tpng tree.dot > tree.png

    print      
    time2 = time.time()
    print 'current time =', time2-time1 , 'sec'
    print
    
    return
        

def RF( TREE_N, FEATURE_M, MAX_DEPTH, CRITERION ):
    
    global X_train, X_test, y_train, y_test, features
    
    # built a RF classifier
    forest_classifier = RandomForestClassifier( n_estimators=TREE_N,
                                                max_features=FEATURE_M,
                                                max_depth = MAX_DEPTH,
                                                random_state=1,
                                                criterion=CRITERION )
    # apply to training data
    forest_classifier = forest_classifier.fit(X_train,y_train)
    
    # predict testing data
    print 'first ten prediction:   ', forest_classifier.predict( X_test )[:10]
    print 'first ten actual value: ', y_test[:10]
    
    # print performance
    report = metrics.precision_recall_fscore_support( y_test,forest_classifier.predict( X_test ),
                                                      average='macro' )
    print 'accuracy: ',forest_classifier.score( X_test, y_test )
    print 'precision: ', report[0]
    print 'recall:    ', report[1]
    print 'fscore:    ', report[2]
    
    # print first ten important valuables
    importances = list(forest_classifier.feature_importances_)
    feature_importances = [ (feature, round(importance, 2)) for feature,
                            importance in zip(features, importances) ]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    
    for pair in feature_importances[:10]:
        print 'Variable: {:<20} Importance: {}'.format(*pair)
    
    return
    
    

# =============================================================================
# main code
# =============================================================================
    
time1 = time.time()



# ===== 1. set up data =====

### set feature names
target_feature = '土壤形態'
features = [ '母質種類', '土壤特性', '排水等級',
             '石灰性', '坡度',
             '表土酸鹼性','第一層質地','第二層質地','第三層質地','第四層質地'     ]
### set data
soil_X = []
soil_y = []
target_feature_values = []

### read file
ReadFile()
# deleted y = 0

### hot encoding
HotEncoding()
# the performance is not so worse without hot encoding

### set training and testing data
X_train, X_test, y_train, y_test = [], [], [], []
TEST_SIZE = 0.1

### split data
SplitData(TEST_SIZE)





# ===== 2. decision tree =====

print
print 'training with DT ...'
print

### set parameters
CRITERION = 'gini'#'entropy'
MAX_DEPTH = None
# no big difference between gini and entropy
# the performance is extremely good without depth restriction

### print parameters    
print 'criterion: ', CRITERION
print 'max_depth: ', MAX_DEPTH

### train data
DT(MAX_DEPTH,CRITERION)



# ===== 3. random forest =====

print
print 'training with RF ...'
print

### set parameters
CRITERION = 'gini'#'entropy'
MAX_DEPTH = None
FEATURE_M = int(np.sqrt(len(features)))
# no big difference between gini and entropy
# the performance is extremely good without depth restriction

### print parameters
print 'criterion: ', CRITERION
print 'num of tree: ', 100
print 'max_features: ', FEATURE_M
print 'max_depth: ', MAX_DEPTH

### train data
RF( 100, FEATURE_M, MAX_DEPTH, CRITERION )



# test different tree number
#for i in [1,2,3,5,7,10,20,50,100,200,500,1000]:
#    print i, ' trees'
#    RF( i, FEATURE_M, MAX_DEPTH, CRITERION )

# test different tree depth
#for i in [1,2,3,5,7,9,10,13,15,20,30,50]:
#    print 'max depth', i
#    RF( 100, FEATURE_M, i, CRITERION )

# test different feature number
#for i in [1,2,3,5,7,9,10,13,15,20,30,50,70,100]:
#    print i, 'features'
#    RF( 100, i, MAX_DEPTH, CRITERION )




print      
time2 = time.time()
print 'done! time =', time2-time1 , 'sec'
print


# ref:
# https://scikit-learn.org
# https://medium.com/@yanweiliu/python機器學習筆記-六-使用scikit-learn建立隨機森林-af13a493f36d
# https://adataanalyst.com/scikit-learn/decision-trees-scikit-learn/
# https://medium.com/@PatHuang/初學python手記-3-資料前處理-label-encoding-one-hot-encoding-85c983d63f87
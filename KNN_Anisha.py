
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import mode
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from math import log10, floor
from metric_learn import LMNN
from sklearn.metrics.pairwise import euclidean_distances
import time
import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE



# In[2]:

# ---- Functions ---- #


# In[3]:

# -- Generate Subsets -- #
def gen_subset (d, train):
    #subs = [];
    subs = np.zeros(shape = [5, np.int(d/100 * train.shape[0]), train.shape[1]])
    subs[0,:] = ( train[np.random.choice(train.shape[0], np.int(d/100 * train.shape[0]), replace = False)] );
    subs[1,:] = ( train[np.random.choice(train.shape[0], np.int(d/100 * train.shape[0]), replace = False)] );
    subs[2,:] = ( train[np.random.choice(train.shape[0], np.int(d/100 * train.shape[0]), replace = False)] );
    subs[3,:] = ( train[np.random.choice(train.shape[0], np.int(d/100 * train.shape[0]), replace = False)] );
    subs[4,:] = ( train[np.random.choice(train.shape[0], np.int(d/100 * train.shape[0]), replace = False)] );
    return subs    


# In[4]:

# ---- LMNN try catch block ---- #
def lmnn_fit_fn(data, k_inp):
    lmnn = lmnn_arr[[np.where(k == k_inp)][0][0][0]]       
    
    try:
        #lmnn = LMNN(k = k_inp, learn_rate=1e-6)
        lmnn.fit(data[:,1:], np.reshape(data[:,0], (1,-1)),  verbose = False)
        data_tx = lmnn.transform(data[:,1:])
    except AssertionError:
        data_tx = data[:,1:]
    except IndexError:
        data_tx = lmnn.transform(data[:,1:]);

    return data_tx;


# In[5]:

# -- cross validation -- #
def CV (train, test, k_inp):
    #LMNN
    # Fit Train data to lmnn
    train_tx = lmnn_fit_fn(train, k_inp)       
    
    # Fit test data to lmnn
    test_tx = lmnn_fit_fn(test, k_inp)
            
    y_train = train[:,0]
    y_test = test[:,0]
           
    acc = knn_fn(train_tx, test_tx, y_train, y_test, k_inp)
    
    return acc;


# In[6]:

# ----- KNN ----- #
def knn_fn (train_mat, test_mat, y_train, y_test, k_inp):
    
    knn_fn.counter += 1;

    euc_dist = np.zeros(test_mat.shape[0])
    y_new = np.zeros(test_mat.shape[0])    
    for i in range (test_mat.shape[0]):
        euc_dist = euclidean_distances(train_mat, np.reshape(test_mat[i], (1,-1)))        
        #np.reshape(test_mat[i], (1,-1)))        
        if euc_dist.shape[0] > k_inp :            
            dist_sort = np.argpartition(euc_dist.T, k_inp - 1)            
            y_new[i] = mode(y_train[dist_sort[0,0:k_inp]])[0] #K NN            
        else:
            y_new[i] = mode(y_train)[0]            
            #dist_sort = np.arange(euc_dist.shape[0]);
            #np.argpartition(euc_dist.T, euc_dist.shape[0] - 1) #.T  transpose
#    print(np.shape(y_new))
#    print(np.shape(y_test))
    accuracy_table = accuracy_score(y_test, y_new)      
    return accuracy_table
knn_fn.counter = 0;


# In[ ]:

# ------ Main Code ------ #


# In[ ]:

#Select input dataset input_dataset = 1 for wine, input_dataset = 2 for MNIST, input_dataset = 3 for office dataset
input_dataset = 1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print('Loading Data ... ', st)

## import inputs wine dataset
fwine = open('wine.data', 'r')
wineraw = np.loadtxt("wine.data", comments="#", delimiter=",", unpack=False) #178x14
wineclass = wineraw[:,1]
winedata = wineraw[:,1:len(wineraw[2,:])]

## import MNIST Dataset
if (input_dataset ==2):
    ftrain = open('mnisttrain.csv', 'r')
    ftest = open('mnisttest.csv','r')
    mnist_trainraw = np.loadtxt("mnisttrain.csv", comments="#", delimiter=",", unpack=False) #785xsamp
    mnist_testraw = np.loadtxt("mnisttest.csv", comments="#", delimiter=",", unpack=False) #785xsamp
    #mnist_raw = np.append(mnist_trainraw,mnist_testraw,axis = 1)
    #mnist_raw = np.roll(np.transpose(mnist_raw),1,axis=1)
    
if (input_dataset == 3):
    ftrain = open('officetrain.csv', 'r')
    ftest = open('officetest.csv','r')
    off_trainraw = np.loadtxt("officetrain.csv", comments="#", delimiter=",", unpack=False) #785xsamp
    off_testraw = np.loadtxt("officetest.csv", comments="#", delimiter=",", unpack=False) #785xsamp
    #off_raw = np.append(mnist_trainraw,mnist_testraw,axis = 1)
    #off_raw = np.roll(np.transpose(mnist_raw),1,axis=1)
    
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print('done', st)


# In[ ]:

# ---- Splitting Data to test train ---- #
if (input_dataset == 1):
    train, test = train_test_split(wineraw, train_size = 0.80)
elif (input_dataset == 2):
    train = np.roll(np.transpose(mnist_trainraw),1,axis=1)
    test = np.roll(np.transpose(mnist_testraw),1,axis=1)              
elif (input_dataset == 3):
    train = np.roll(off_trainraw,1,axis=1)
    test = np.roll(off_testraw,1,axis=1)
    train = train[:,0:train.shape[1]-2]
    test = test[:,0:test.shape[1]-2]


# In[ ]:

#if (input_dataset == 1):
# ----- LDA ----- #
lda_model = LDA()
temp_data = lda_model.fit_transform(train[:,1:], train[:,0])
train = np.append(np.reshape(train[:,0], (-1,1)), temp_data, axis = 1 )

temp_data = lda_model.transform(test[:,1:])
test = np.append(np.reshape(test[:,0], (-1,1)), temp_data, axis = 1 )

#elif (input_dataset == 2):
#    # ----- Mnist Dataset ----- #
#    tsne_model = TSNE();
#    temp_data = tsne_model.fit_transform(train[:,1:])
#    train = np.append(np.reshape(train[:,0], (-1,1)), temp_data, axis = 1 )
    
#    tsne_model = TSNE()
#    temp_data = tsne_model.fit_transform(test[:,1:])
#    test = np.append(np.reshape(test[:,0], (-1,1)), temp_data, axis = 1 )


# In[ ]:

# ---- Randomly select a subset of d = (20; 50; 80; 100) ----- #
subsets20 = np.zeros(shape = [5, np.int(20/100 * train.shape[0]), train.shape[1]])
subsets50 = np.zeros(shape = [5, np.int(50/100 * train.shape[0]), train.shape[1]])
subsets80 = np.zeros(shape = [5, np.int(80/100 * train.shape[0]), train.shape[1]])

subsets20 = gen_subset(20,train)
subsets50 = gen_subset(50,train)
subsets80 = gen_subset(80,train)

print('done')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print('Data Loaded ', st)


# In[ ]:

k = np.array([3,5,7])
d_val = 3
lmnn_arr = []
subset_val = 5

if (input_dataset == 1):
    fold_val = 3 # includes leave one out for wine dataset
else:
    fold_val = 2        

for k_it in range(0,k.shape[0],1):
    lmnn_arr.append( LMNN(k = k[k_it], learn_rate=1e-6))


# In[ ]:

# ---- Finding K fold train and test indices ----- #
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print('Computing Accuracy matrix ',st)

        
accuracy_mat = np.zeros(shape = [d_val,subset_val,fold_val,k.shape[0]]) #d,subset,fold, k. #2fold = 0, #5fold = 1 #LOO = 2.

for d_it in range(0,3,1):
    print('d ',d_it)
    if (d_it == 0):
        subsets = subsets20
    elif (d_it == 1):
        subsets = subsets50        
    elif (d_it == 2):
        subsets = subsets80   
         
    for sub_it in range(0,5,1): 
        print('subset ',sub_it)
        
        #@ 2 fold Cross Validation
        kf = KFold(subsets[sub_it].shape[0], n_folds=2)
        print('2 Fold')
        for k_it in range(0,k.shape[0],1):           
            print('k ',k_it)
            acc_sum= 0;            
            for train_index, test_index in kf:
                acc = CV(subsets[sub_it][train_index, :], subsets[sub_it][test_index,:], k[k_it])
                acc_sum = acc_sum + acc;
            accuracy_mat[d_it,sub_it,0,k_it] = acc_sum/2;                        
        
        ## 5 fold Cross Validation
        kf = KFold(subsets[sub_it].shape[0], n_folds=5)
        print('5 Fold')
        for k_it in range(0,k.shape[0],1):
            print('k ',k_it)
            acc_sum= 0;
            for train_index, test_index in kf:
                acc = CV(subsets[sub_it][train_index, :], subsets[sub_it][test_index,:], k[k_it])
                acc_sum = acc_sum + acc;            
            accuracy_mat[d_it,sub_it,1,k_it] = acc_sum/5;

        if(input_dataset == 1):
            #Leave one out fold Cross Validation
            print('Leave one out')
            kf = KFold(subsets[sub_it].shape[0], n_folds=subsets[sub_it].shape[0])
            for k_it in range(0,k.shape[0],1): 
                print('k ',k_it)
                acc_sum= 0;
                for train_index, test_index in kf:
                    acc = CV(subsets[sub_it][train_index, :], subsets[sub_it][test_index,:], k[k_it])
                    acc_sum = acc_sum + acc;            
                accuracy_mat[d_it,sub_it,2,k_it] =  acc_sum/subsets[sub_it].shape[0];

print('subset 100%',)            
# for d = 100%
accuracy_mat_100d = np.zeros(shape = [fold_val,k.shape[0]]) #d,subset,fold, k. #2fold = 0, #5fold = 1 #LOO = 2.
#@ 2 fold Cross Validation
kf = KFold(subsets[sub_it].shape[0], n_folds=2)
print('2 Fold')
for k_it in range(0,k.shape[0],1): 
    print('k ',k_it)
    acc_sum= 0;            
    for train_index, test_index in kf:
        acc = CV(subsets[sub_it][train_index, :], subsets[sub_it][test_index,:], k[k_it])
        acc_sum = acc_sum + acc;
    accuracy_mat_100d[0,k_it] = acc_sum/2;                        

## 5 fold Cross Validation
kf = KFold(subsets[sub_it].shape[0], n_folds=5)
print('5 Fold')
for k_it in range(0,k.shape[0],1): 
    print('k ',k_it)
    acc_sum= 0;
    for train_index, test_index in kf:
        acc = CV(subsets[sub_it][train_index, :], subsets[sub_it][test_index,:], k[k_it])
        acc_sum = acc_sum + acc;            
    accuracy_mat_100d[1,k_it] = acc_sum/5;


if (input_dataset == 1):
    #Leave one out fold Cross Validation
    kf = KFold(subsets[sub_it].shape[0], n_folds=subsets[sub_it].shape[0])
    print('Leave one out')
    for k_it in range(0,k.shape[0],1): 
        print('k ',k_it)
        acc_sum= 0;
        for train_index, test_index in kf:
            acc = CV(subsets[sub_it][train_index, :], subsets[sub_it][test_index,:], k[k_it])
            acc_sum = acc_sum + acc;            
        accuracy_mat_100d[2,k_it] =  acc_sum/subsets[sub_it].shape[0];


print('done')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print('Accuracy matrix generated ',st)


# In[ ]:

# ---- Finding best K ---- #
acc_sum_k = np.zeros(shape = k.shape[0])
acc_sum_100d = np.zeros(shape = k.shape[0])
for k_int in range(0,k.shape[0],1):
    acc_sum_k[k_int] = np.sum (accuracy_mat[:,:,:,k_int])
    acc_sum_100d[k_int] = np.sum(accuracy_mat_100d[:,k_int])

den = (d_val * subset_val * fold_val) + (fold_val)
acc_means_k = np.zeros(shape = k.shape[0])
acc_mean_k = (np.add(acc_sum_k, acc_sum_100d))/den

print('Accuracy with respect to k = ',acc_mean_k )
    
k_best = k[np.argmax(acc_means_k)]

# ----- Applying the best K to test data and finding final accuracy ----- #

# Fit train data to lmnn
train_tx = lmnn_fit_fn(train, k_best)

# Fit test data to lmnn
test_tx = lmnn_fit_fn(test, k_best)


acc_best_k = knn_fn(train_tx, test_tx, train[:,0], test[:,0], k_best)
print('Best K = ',k_best)
print('Final Accuracy Score = ',acc_best_k)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print('Execution end time ',st)


# In[ ]:

#for k_best in k:
    # Fit train data to lmnn
train_txa = lmnn_fit_fn(train, k_best)

# Fit test data to lmnn
test_txa = lmnn_fit_fn(test, k_best)

acc_best_k = knn_fn(train_txa, test_txa, train[:,0], test[:,0], k_best)
print('Best K = ',k_best)
print('Final Test Accuracy Score = ',acc_best_k)
print('Test Error percentage for best k = ', (1-acc_best_k)*100,'%')

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print('Execution end time ',st)


# In[ ]:

lmnn_arr = []

k = np.array([1,3,5])
        
for k_it in range(0,k.shape[0],1):
    lmnn_arr.append( LMNN(k = k[k_it], learn_rate=1e-6))
        

#%%

#Cross_Validation accuracy
cv_acc = (np.sum(accuracy_mat[:,:,:,np.where(k==k_best)[0][0]]) + np.sum(accuracy_mat_100d[:,np.where(k==k_best)[0][0]])) /(d_val* subset_val * fold_val + fold_val) 

print('CV accuracy', cv_acc)
print('Cross Validation Error = ', (1-cv_acc)*100,'%')
# In[ ]:

# Which F worked best
acc_sum_f = np.zeros(shape = fold_val)
acc_sum_100d_f = np.zeros(shape = fold_val)
for f_int in range(0,fold_val,1):
    acc_sum_f[f_int] = np.sum (accuracy_mat[:,:,f_int,:])
    acc_sum_100d_f[f_int] = np.sum(accuracy_mat_100d[f_int,:])

den = (d_val * subset_val * k.shape[0]) + (k.shape[0])
acc_means_f = np.zeros(shape = fold_val)
acc_mean_f = (np.add(acc_sum_f, acc_sum_100d_f))/den


print('Accuracy with respect to 2 Fold, 5 fold', acc_mean_f )
# In[ ]:


#Statbility over subset of same size

acc_sum_sub = np.zeros(shape = [d_val, subset_val])
for d_int in range(0,d_val,1):
    for sub_int in range(0,subset_val,1):
        acc_sum_sub[d_int,sub_int] = np.sum(accuracy_mat[d_int,sub_int,:,:])  
        

den = (fold_val * k.shape[0])
acc_sum_sub = acc_sum_sub/den

acc_var_sub = np.zeros(shape = 3)
acc_var_sub[0] = np.var(acc_sum_sub[0,:])
acc_var_sub[1] = np.var(acc_sum_sub[1,:])
acc_var_sub[2] = np.var(acc_sum_sub[2,:])        

print('Variance in accuracy of between the subsets of same size is ', acc_var_sub)

# In[ ]:

acc_mean_d = np.mean(acc_sum_sub, axis = 1)

print('Accuracy with respect to the size of the training subset', acc_mean_d)
# In[ ]:




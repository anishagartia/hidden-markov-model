# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:11:37 2016

@author: gartiya
"""
import numpy as np

ftrain = open('robot_train.data','r')
ftest = open('robot_test.data','r')
i = 0;
j = 0;
k = 0;
 
train =  [[[0 for _ in range(3)] for _ in range(200)] for _ in range(200)]
test =  [[[0 for _ in range(3)] for _ in range(200)] for _ in range(200)]
# train shape = 200,200,3 ie datapoints, obs, (coord, color)


for line in ftrain:
    for row in line:        
        if (row == ':'):            
            continue;            
        elif (row == ' '):            
            continue;            
        elif (row == '\n'):
            j = j +1;
            k = 0;
            break;
        elif (row == '.'):                
            i = i + 1
            j = -1;
            k = 0;            
            continue;
        else:           
            train[i][j][k] = row                                           
            k = k+1;                                                                  

i = 0;
j = 0;
k = 0;
for line in ftest:
    for row in line:        
        if (row == ':'):            
            continue;            
        elif (row == ' '):            
            continue;            
        elif (row == '\n'):
            j = j +1;
            k = 0;
            break;
        elif (row == '.'):                
            i = i + 1
            j = -1;
            k = 0;            
            continue;
        else:           
            test[i][j][k] = row                                           
            k = k+1; 


# initialize all structures

# Initialize A and prior
# Index : (1,1) (1,2) .. (1,4) (2,1)...(2,4) (3,1)..(3,4) (4,1)..(4,4)
n= 16;
v = 4;
t = 200;
o = 200;
alpha = np.zeros(shape = [t,0,n])
beta = np.zeros(shape = [n,t])
gamma = np.zeros([])
epsilon = np.zeros([])
A = np.zeros(shape = [n,n])
B = np.zeros(shape = [n,v])
prior = np.zeros(shape = [1,n])
states_occ = np.ones(shape = [n,1])


for i in range(0,t,1):
    for j in range(0,o-1,1):              
                A[((int(train[i][j][0])-1)*4 + int(train[i][j][1]) - 1 ), ((int(train[i][j+1][0])-1)*4 + int(train[i][j+1][1]) - 1 )] += 1
                states_occ[ ((int(train[i][j][0])-1)*4 + int(train[i][j][1]) - 1 ) , 0] += 1            
                if (j == 0):
                    prior[0, ((int(train[i][j][0])-1)*4 + int(train[i][j][1]) - 1 )  ] += 1                    

states_occ = np.repeat(states_occ, n , axis = 1)
A = A/states_occ;
prior = prior/(t)

states_occ = np.ones(shape = [n,1])
# Initialize B
# B index of symbols r g b y
for i in range(0,t,1):
    for j in range(0,o,1):
       
       given_color = train[i][j][2]
       if (given_color == 'r'):
           col = 0
       elif (given_color == 'g'):
           col = 1
       elif (given_color == 'b'):
           col = 2
       elif (given_color == 'y'):
           col = 3
        
       B[ ((int(train[i][j][0]) -1)*4 + (int(train[i][j][1]) - 1)) , col ] += 1;
       states_occ[ ((int(train[i][j][0]) -1)*4 + (int(train[i][j][1]) - 1)) , 0 ] += 1;
       
states_occ = np.repeat(states_occ, v , axis = 1)        
B = B/(states_occ)


train_tx = np.zeros(shape = [t,o,2])
test_tx = np.zeros(shape = [t,o,2])
# ---- Changing colour and state in train matrix  to int---- #
# r = 0, g = 1, b = 2, y - 3
for i in range(0,t,1):
    for j in range(0,o,1):
    
       given_color = train[i][j][2]
       
       if (given_color == 'r'):
            col = '0'
       elif (given_color == 'g'):
           col = '1'
       elif (given_color == 'b'):
           col = '2'
       elif (given_color == 'y'):
           col = '3'
           
       train_tx[i,j,0] = (int(train[i][j][0]) -1)*4 + (int(train[i][j][1]) - 1)            
       train_tx[i,j,1] = col
       
       
       given_color = test[i][j][2]
       
       if (given_color == 'r'):
            col = '0'
       elif (given_color == 'g'):
           col = '1'
       elif (given_color == 'b'):
           col = '2'
       elif (given_color == 'y'):
           col = '3'
           
       test_tx[i,j,0] = (int(test[i][j][0]) -1)*4 + (int(test[i][j][1]) - 1)            
       test_tx[i,j,1] = col

# ---- Initialization ---- #
alpha = np.zeros(shape = [t,o,n])
q = np.array(train_tx[:,0,1])
biO = B[ : , q.astype(int) ]
alpha[:,0,:] = prior * biO.T


# ---- Induction ---- #
for o_int in range(0,o-1,1):
    P = (alpha[:,o_int,:] @ A[:,:]) #200x16
    Q = B[:,train_tx[:,o_int,1].astype(int)] #16x200
    alpha[:,o_int +1 ,:] = P*Q.T

Prob_seq = np.zeros(shape = [t,1])
Prob_seq[:,0]  = np.sum(alpha[:,o-1,:], axis = 1)


# ------ VITERBI for TEST------ #
# Creaing data structures
delta = np.zeros(shape = [t,o,n])
psi = np.zeros(shape = [t,o,n])
P_star = np.zeros(shape = [t,1])
q_star = np.zeros(shape = [t,o])

# Initialization
q = np.array(test_tx[:,0,1])
biO = B[ : , q.astype(int) ]
delta[:,0,:] = prior * biO.T

#Recursion
for t_int in range(0,t,1):
    for o_int in range(1,o,1):        
        P_t_int = np.max(delta[t_int,o_int-1,:] * A, axis = 1)#16x1#.reshape(1,-1) #1x16              
        Q = B[:,test_tx[t_int,o_int,1].astype(int)] #16x1
        delta[t_int,o_int,:] = P_t_int * Q #16x1
        
        psi[t_int, o_int,:] = np.argmax( delta[t_int, o_int-1 ,:] * A, axis = 1).reshape(1,-1) #1x16 
           
# Termination
P_star = np.max(delta[:,o-1,:], axis = 1)
q_star[:,o-1] = np.argmax(delta[:,o-1,:], axis = 1)

# Path Backtracking
for t_int in range(0,t,1):
    for o_int in range(o-2,-1,-1):        
        #q_temp = q_star[:,o_int+1].astype(int)
        q_star[t_int,o_int] = psi[t_int,o_int+1,q_star[t_int,o_int+1].astype(int)]
        
        
#%% 
#Visualize the observation probabilites    
obs = np.empty(size = [4,4,4])
for i in range(0,4,1):
    for j in range(0,4,1):
        obs[i,j,:] = B[(i)*4 + j,:]

#%%
#average error across all test sequences
error_mat = np.zeros(shape = [200,1])

for i in range(0,200,1):
    for j in range(0,200,1):
        if (q_star[i,j] != test_tx[i,j,0]):
            error_mat[i,0] += 1            
error_mat = error_mat/200
#%%





#%%





#%%





#%%





#%%
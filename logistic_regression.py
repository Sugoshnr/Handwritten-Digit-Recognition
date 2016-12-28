import pickle
import gzip
import numpy as np
import math
import sys
import time
#import h5py
start =time.time()
def design_T(train_y,validation_y,test_y):
    T_train=np.zeros((len(train_y),10))
    T_validation=np.zeros((len(validation_y),10))
    T_test=np.zeros((len(test_y),10))

    for i in range(len(train_y)):
        T_train[i][train_y[i]]=1
    for i in range(len(validation_y)):
        T_validation[i][validation_y[i]]=1
    for i in range(len(test_y)):
        T_test[i][test_y[i]]=1

    return T_train,T_validation,T_test



filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f)
f.close()

train_x_temp=training_data[0]
train_y=training_data[1]

validation_x_temp=validation_data[0]
validation_y=validation_data[1]

test_x_temp=test_data[0]
test_y=test_data[1]

# hf=h5py.File('USPS.h5','r')
# usps_x = (255-255*np.asarray(hf.get('features')))/255
# usps_x = 1-np.asarray(hf.get('features'))
# usps_y=np.asarray(hf.get('target'))



# train_x=np.zeros((len(train_x_temp),len(train_x_temp[0])+1))
# validation_x=np.zeros((len(validation_x_temp),len(validation_x_temp[0])+1))
# test_x=np.zeros((len(test_x_temp),len(test_x_temp[0])+1))


# for i in range(len(train_x_temp)):
#     train_x[i]=np.append(0,train_x_temp[i])
# for i in range(len(validation_x_temp)):
#     validation_x[i]=np.append(0,validation_x_temp[i])
# for i in range(len(test_x_temp)):
#     test_x[i]=np.append(0,test_x_temp[i])


train_x=train_x_temp
validation_x=validation_x_temp
test_x=test_x_temp

W=np.ones((10,len(train_x[0])))

Y=np.zeros(10)

T_train, T_validation, T_test=design_T(train_y,validation_y,test_y)
print ("STARTED")
ETerm=np.zeros(W.shape)
cnt=0
for z in range(5):
    print ("ITERATION "+str(z+1))
    for i in range(len(train_x)):
        A=np.dot(W,train_x[i])+1
        denom = np.sum(np.exp(A))
        E=0
        for k in range(10):
            Y[k]=np.exp(A[k])/denom
            E-=np.dot(T_train[i][k],np.log(Y[k]))
        cnt+=1
        for j in range(10):
            W[j]=W[j]-0.01*(Y[j]-T_train[i][j])*train_x[i]
        # Uncomment for Mini batch stochastic gradient descent
        #    ETerm[j]+=((T_train[i][j]-Y[j])*train_x[i])
        #if(cnt%10==0):
        #    W=W+0.01*ETerm/10
        #    ETerm=np.zeros(W.shape)
    print (E)

count=0.0
for i in range(len(train_x)):
    A=np.dot(W,train_x[i])
    denom = np.sum(np.exp(A))
    for k in range(10):
        Y[k]=np.exp(A[k])/denom
    if np.where(Y==max(Y))[0][0]==np.where(T_train[i]==max(T_train[i]))[0][0]:
        count+=1
print ("TRAINING ACCURACY")
print (count/len(train_x))

count=0.0
for i in range(len(validation_x)):
    A=np.dot(W,validation_x[i])
    denom = np.sum(np.exp(A))
    for k in range(10):
        Y[k]=np.exp(A[k])/denom
    if np.where(Y==max(Y))[0][0]==np.where(T_validation[i]==max(T_validation[i]))[0][0]:
        count+=1
print ("VALIDATION ACCURACY")
print (count/len(validation_x))

count=0.0
for i in range(len(test_x)):
    A=np.dot(W,test_x[i])
    denom = np.sum(np.exp(A))
    for k in range(10):
        Y[k]=np.exp(A[k])/denom
    if np.where(Y==max(Y))[0][0]==np.where(T_test[i]==max(T_test[i]))[0][0]:
        count+=1
print ("TESTING ACCURACY")
print (count/len(test_x))

print ("Running time = "+str(time.time()-start))

# for i in range(len(usps_x)):
#    A=np.dot(W,usps_x[i])
#    denom = np.sum(np.exp(A))
#    for k in range(10):
#        Y[k]=np.exp(A[k])/denom
#    if np.where(Y==max(Y))[0][0]==usps_y[i]:#np.where(T_test[i]==max(T_test[i]))[0][0]:
#        count+=1
# print ("USPS")
# print (count/len(usps_x))

        


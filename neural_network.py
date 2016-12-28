import pickle
import gzip
import numpy as np
import math
import sys
import time
#import h5py

start=time.time()
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



def activate(Z):
    #Sigmoid
    out=np.zeros(len(Z))
##    for i in range(len(Z)):
##        out[i]=1.0/(1.0+math.exp(-Z[i]))
##    #tanh
##    for i in range(len(Z)):
##        out[i]=(np.exp(2*Z[i])-1)/(np.exp(2*Z[i])+1)
##
##    #Relu
    for i in range(len(Z)):
        out[i]=max(0,Z[i])


    return out


def derivative(Z):
    out=np.zeros(Z.shape)
    #Sigmoid
##    out=Z*(1-Z)
    #tanh
##    out=(1-Z**2)
    #Relu
    for i in range(len(Z)):
        if(Z[i]>0):
            out[i]=1
    return out
    
   
def softmax(Z):
    out=np.zeros(len(Z))
    denom=np.sum(np.exp(Z))
    for i in range(len(Z)):
        out[i]=np.exp(Z[i])/denom
    return out

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

train_x=np.zeros((len(train_x_temp),len(train_x_temp[0])+1))
validation_x=np.zeros((len(validation_x_temp),len(validation_x_temp[0])+1))
test_x=np.zeros((len(test_x_temp),len(test_x_temp[0])+1))


for i in range(len(train_x_temp)):
    train_x[i]=np.append(0,train_x_temp[i])
for i in range(len(validation_x_temp)):
    validation_x[i]=np.append(0,validation_x_temp[i])
for i in range(len(test_x_temp)):
    test_x[i]=np.append(0,test_x_temp[i])


W1=np.random.rand(100,785)
W1=W1*2*0.12
W1=W1-0.12
W2=np.random.rand(10,101)
W2=W2*2*0.12
W2=W2-0.12


Y=np.zeros(10)


# hf=h5py.File('USPS.h5','r')
# usps_x_temp = 1-np.asarray(hf.get('features'))
# usps_y=np.asarray(hf.get('target'))
# usps_x=np.zeros((len(usps_x_temp),len(usps_x_temp[0])+1))
# for i in range(len(usps_x_temp)):
#    usps_x[i]=np.append(0,usps_x_temp[i])


T_train, T_validation, T_test=design_T(train_y,validation_y,test_y)
print ("TRAINING STARTED")
for k in range(5):
    cnt=0
    #ETerm1=np.zeros(W1.shape)
    #ETerm2=np.zeros(W2.shape)
    print ("ITERATION "+str(k+1))
    for i in range(len(train_x)):
        E=0
        Z_2=np.dot(W1,train_x[i])
        A=activate(Z_2)
        A_new=np.append(1,A)
        Z_3=np.dot(W2,A_new)
        Y=softmax(Z_3)
        delta_k=Y-T_train[i]
        temp=np.dot(W2.T,delta_k)
        derivative_val=derivative(A_new)
        delta_j=np.multiply(derivative_val,temp)
        #cnt+=1
        W1=W1-0.01*np.asarray(np.asmatrix(delta_j[1:]).T*np.asmatrix(train_x[i]))
        W2=W2-0.01*np.asarray(np.asmatrix(delta_k).T*np.asmatrix(A_new))
        # Uncomment for mini batch stochastic gradient descent
        #ETerm1+=np.asarray(np.asmatrix(delta_j[1:]).T*np.asmatrix(train_x[i]))
        #ETerm2+=np.asarray(np.asmatrix(delta_k).T*np.asmatrix(A_new))
        #if cnt%10==0:
        #    W1=W1-0.01*ETerm1/10
        #    W2=W2-0.01*ETerm2/10
        #    ETerm1=np.zeros(W1.shape)
        #    ETerm2=np.zeros(W2.shape)
        for k in range(10):
            E-=np.dot(T_train[i][k],np.log(Y[k]))
    print (E)
count=0.0
for i in range(len(train_x)):

    Z_2=np.dot(W1,train_x[i])
    A=activate(Z_2)
    A_new=np.append(1,A)
    Z_3=np.dot(W2,A_new)
    Y=softmax(Z_3)
    if( np.argmax(Y) == np.argmax(T_train[i])):
        count+=1

print ("TRAINING ACCURACY")
print (count/len(train_x))

count=0.0
for i in range(len(validation_x)):
    Z_2=np.dot(W1,validation_x[i])
    A=activate(Z_2)
    A_new=np.append(1,A)
    Z_3=np.dot(W2,A_new)
    Y=softmax(Z_3)
    if( np.argmax(Y) == np.argmax(T_validation[i])):
        count+=1

print ("VALIDATION ACCURACY")
print (count/len(validation_x))

count=0.0
for i in range(len(test_x)):

    Z_2=np.dot(W1,test_x[i])
    A=activate(Z_2)
    A_new=np.append(1,A)
    Z_3=np.dot(W2,A_new)
    Y=softmax(Z_3)
    if( np.argmax(Y) == np.argmax(T_test[i])):
        count+=1

print ("TESTING ACCURACY")
print (count/len(test_x))

print ("Running time = "+str(time.time()-start))
# count=0.0
# for i in range(len(usps_x)):

#    Z_2=np.dot(W1,usps_x[i])
#    A=activate(Z_2)
#    A_new=np.append(1,A)
#    Z_3=np.dot(W2,A_new)
#    Y=softmax(Z_3)
#    if( np.argmax(Y) == usps_y[i]):
#        count+=1

# print ("USPS")
# print (count/len(usps_x))


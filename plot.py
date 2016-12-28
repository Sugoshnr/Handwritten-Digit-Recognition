from matplotlib import pyplot as plt
import numpy as np



f=open("NN_sgd.txt","r")
f1=open("NN_bsgd.txt","r")
x=[]
y1=[]
y2=[]
y3=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))

for line in f1:
    X=line.split()
    y2.append(float(X[0]))
    y3.append(float(X[1]))

plt.figure(1)
plt.title("Error values")
plt.xlabel("Iterations")
plt.ylabel("Cross-Entropy")
a,=plt.plot(x,y1,label="Stochastic Gradient Descent")
b,=plt.plot(y2,y3,label="Mini-Batch Stochastic Gradient Descent of size 10")
plt.legend(handles=[a,b])
#for a,b in zip(x, y): 
#    plt.text(a, b, str(b))
plt.savefig("NN")


plt.show()


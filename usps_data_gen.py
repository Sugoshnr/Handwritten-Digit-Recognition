import cPickle
import gzip
import numpy as np
import math
import sys
import cv2
import h5py


#PATH="H:\\CSE574-Introduction_to_Machine_Learning\\Project_3\\Numerals\\"+str(i)+"\\"


##for i in range(3,10):
##        count=1
##        #PATH="/media/sugosh/New Volume/CSE574-Introduction_to_Machine_Learning/Project_3/USPSdata/Numerals/"+str(i)+"/"
##        PATH="H:\\CSE574-Introduction_to_Machine_Learning\\Project_3\\Numerals\\"+str(i)+"\\"
##        for j in range(1,2001):
##                img1=cv2.imread(PATH+"1 ("+str(j)+").png",0)
##                img1=cv2.resize(img1,(28,28))
##                #img1=img1.reshape((28,28))
##                cv2.imwrite(PATH+str(count)+".jpg",img1)
##                count+=1
##                print count
##        print i   


# l1=[]
# l2=[]      
# for i in range(10):
#         PATH="/media/sugosh/New Volume/CSE574-Introduction_to_Machine_Learning/Project_3/Numerals/"+str(i)+"/"
#         for j in range(1,2000):
#                 img1=cv2.imread(PATH+str(j)+".jpg",0)
#                 img1=img1.flatten()/255.0
#                 l1.append(img1)
#                 l2.append(i)

# l1=np.asarray(l1)
# l2=np.asarray(l2)

# with h5py.File('USPS.h5', 'w') as hf:
#     hf.create_dataset('features', data=l1)
#     hf.create_dataset('target', data=l2)

hf=h5py.File('USPS.h5','r')
features = hf.get('features')
    
print features[0]  
# cv2.imshow("",img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
 

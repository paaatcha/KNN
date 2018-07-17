import sys
import numpy as np
from knn import knn

sys.path.append("../utils")
from utils import cont_error, ind2vec

# loading the data set
dataset = np.genfromtxt('../datasets/iris.csv', delimiter=',')


# Number of samples and features + label (the last position of the array is the class label)
[nsp, feat] = dataset.shape


nIter = 30
miss = list()
k = 11

for it in range(nIter):    
    # Shuffling the dataset
    np.random.shuffle(dataset)
    
    # Getting 70% for training and 30% for tests
    sli = int(round(nsp*0.7))
    in_train = dataset[0:sli,0:feat-1]
    out_train = ind2vec((dataset[0:sli,feat-1])-1)
    in_test = dataset[sli:nsp,0:feat-1]
    out_test = ind2vec(dataset[sli:nsp,feat-1]-1)
    
    res = knn (in_train, out_train, in_test, k)
#    print res
    acc = ((len(in_test) - cont_error (out_test, res))/45.0)*100
    miss.append(acc)
    print 'number of missclassification: ', acc

miss = np.asarray(miss)
print 'AVG: ', miss.mean(), '\nSTD: ', miss.std()







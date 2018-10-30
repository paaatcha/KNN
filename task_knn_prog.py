import numpy as np
from knn import knn
import sys
sys.path.append('/home/patcha/Dropbox/Doutorado/Codigos/Python/utils')
from utilsClassification import oneHotEncoding, contError
from sklearn.metrics import confusion_matrix

path_dataset_train = '/home/patcha/Prog2018-2/ccomp/bateria_validacao/iris/iris_treino.csv'
path_dataset_test = '/home/patcha/Prog2018-2/ccomp/bateria_validacao/iris/iris_teste.csv'
k = 3
dist = 'C'
r = 1

# loading the data set
dataset_train = dataset = np.genfromtxt(path_dataset_train, delimiter=',')
dataset_test = dataset = np.genfromtxt(path_dataset_test, delimiter=',')
 
in_train = dataset_train[:,0:-1]
out_train = oneHotEncoding((dataset_train[:,-1])-1)
in_test = dataset_test[:,0:-1]
out_test = oneHotEncoding(dataset_test[:,-1]-1)

res = knn (in_train, out_train, in_test, k, dist, r)

n_sam_test = float(len(in_test))

acc = ((n_sam_test - contError (out_test, res))/n_sam_test)*100

print 'number of missclassification: ', acc

cm = confusion_matrix(out_test.argmax(axis=1), res.argmax(axis=1))

print cm















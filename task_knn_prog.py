import numpy as np
from knn import knn
import sys
sys.path.append('/home/labcin/Dropbox/Doutorado/Codigos/Python/utils')
from utilsClassification import oneHotEncoding, contError
from sklearn.metrics import confusion_matrix

path_dataset_train = '/home/labcin/AndrePacheco/TrabalhosProg/Prog2018-2/ccomp/bateria_validacao/vowels/vowels_treino.csv'
path_dataset_test = '/home/labcin/AndrePacheco/TrabalhosProg/Prog2018-2/ccomp/bateria_validacao/vowels/vowels_teste.csv'
dataset_name = path_dataset_train.split('/')[-2]

k = [11, 13, 13, 15, 16, 17, 19, 19, 15, 18]
dist = ['C', 'M', 'E', 'E', 'C', 'M', 'M', 'E', 'M', 'C']
r = [1, 0.5, 1, 1, 1, 0.7, 0.3, 1, 1.5, 1]

if (not (len(k) == len(dist) == len(r))):
    raise ('k, dist and r must have the same lenght')

with open(dataset_name + '/config.txt', 'w') as f:
    f.write('dataset/' + path_dataset_test.split('/')[-1] + '\n')
    f.write('dataset/' + path_dataset_test.split('/')[-1] + '\n')
    f.write('predicoes/\n')
    
    for ki, di, ri in zip (k, dist, r):
        if (di == 'M'):
            s = str(ki) + ', M, ' + str(ri) + '\n'
        else:
            s = str(ki) + ', ' + di + '\n'
        f.write(s)
    

# loading the data set
dataset_train = dataset = np.genfromtxt(path_dataset_train, delimiter=',')
dataset_test = dataset = np.genfromtxt(path_dataset_test, delimiter=',')
 
in_train = dataset_train[:,0:-1]
out_train = oneHotEncoding((dataset_train[:,-1])-1)
in_test = dataset_test[:,0:-1]
out_test = oneHotEncoding(dataset_test[:,-1]-1)
n_sam_test = float(len(in_test))

cont = 1

for ki, di, ri in zip (k, dist, r):

    res = knn (in_train, out_train, in_test, ki, di, ri)
    acc = ((n_sam_test - contError (out_test, res))/n_sam_test) 
    acc = round(acc,2)
    cm = confusion_matrix(out_test.argmax(axis=1), res.argmax(axis=1))
    
    print "k = {}, di = {} and r = {}".format(ki,di,ri)
    print acc
    print cm
    print '\n'
    
    with open(dataset_name + "/predicoes/predicao_" + str(cont) + '.txt', 'w') as f:
        f.write(str(acc) + '\n\n')    
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                f.write(str(cm[i,j]))
                if (j < cm.shape[1]-1):
                    f.write(' ')
            f.write('\n')
            
        f.write('\n')
        for out in res:
            f.write(str(out.argmax()) + '\n')

    cont += 1














import numpy as np,numpy.random
from hmmlearn import hmm
import pandas as pd
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = np.loadtxt("subsitution/ciphertext.txt",dtype=str)
data1 = np.loadtxt("vigenere/mul_vigenere_cipher",dtype=str)
data2 = np.loadtxt("transposition/ciphertext_multikeys",dtype=str)
data3 = np.loadtxt("hill/encryptedText",dtype=str)
header = ['key','plaintxt','ciphertxt','none']
dataset = pd.read_csv('playfair/Playfair.csv',names=header)
data4 = dataset['ciphertxt']

samples = [data,data1,data2,data3,data4]
samples = np.asarray(samples)
index = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n',
         'o','p','q','r','s','t','u','v','w','x','y','z']
train = []
test = []
for data in samples:
    trainsample = []
    for i in data:
        temp = []
        i = i.lower()
        for j in i:
            if j in index:
                num = index.index(j)
                temp.append(num)
        trainsample.append(temp)
    train.append(trainsample[:800])
    test.append(trainsample[800:])

train = np.asarray(train)
test = np.asarray(test)
test = np.concatenate(test,axis=0)
#test = np.ravel(test)
print(train.shape,test.shape)
hidden_state = 2
# symbols = 26
# pi = np.random.dirichlet(np.ones(hidden_state), size=1)
# A = np.random.dirichlet(np.ones(hidden_state), size=hidden_state)
# B = np.random.dirichlet(np.ones(symbols), size=hidden_state)

model = hmm.MultinomialHMM(n_components=hidden_state,init_params='ste', n_iter=10)
# model.startprob_ = pi
# model.transmat_ = A
# model.emissionprob_ = B
score = []
counter = 0
for cipher in train:
    input =np.concatenate(cipher,axis=0)

    print("number of ciphertxt: ",counter)
    hidden_state = 26
    # symbols = len(input)
    # pi = np.random.dirichlet(np.ones(hidden_state), size=1)
    # A = np.random.dirichlet(np.ones(hidden_state), size=hidden_state)
    # B = np.random.dirichlet(np.ones(symbols), size=hidden_state)
    model = hmm.MultinomialHMM(n_components=hidden_state,init_params='ste', n_iter=10,verbose=True)
    # model.startprob_ = pi
    # model.transmat_ = A
    # model.emissionprob_ = B
    input = np.reshape(input,(-1,1))
    model.fit(input)
    for i in test:
        i = np.reshape(i,(-1,1))
        s = model.score(i)
        print(s)
        score.append(s)
    counter = counter+1


# score = np.asarray(score)
# vect = score.reshape(5,1000).T
# print(vect)
np.savetxt("result1.txt",score,fmt="%s")
#print(score.shape)



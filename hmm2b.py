import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import pickle

from utils import read_find_and_convert_to_np_array, convert_array_to_str

ciphers = ['ciphers/column_transposition_cipher',
            'ciphers/hill_cipher',
            'ciphers/playfair_cipher',
            'ciphers/simple_substitution_cipher'
            'ciphers/vigenere_cipher']

c = ['column_transposition', 'hill_cipher', 'playfair_cipher', 'simple_substitution', 'vignere']

models = [None]
for i in range(len(ciphers)):
    for j in range(100):
        print('Training ', ciphers[i])
        model = hmm.MultinomialHMM(n_components=26, init_params="ste", n_iter=2, verbose=True)
        #
        data = np.array(read_find_and_convert_to_np_array(ciphers[i]))
        train_data, test_data = train_test_split(data[50:], test_size=0.2, shuffle=True)
        train = np.concatenate(train_data, axis=0)
        length = [len(train_data[0]), len(train_data[1])]
        #
        model.fit(np.array([train]).T)

        #print('input = ', convert_array_to_str(test_data[0]))
        #result = model.decode(np.array([test_data[0]]).T, algorithm="viterbi")
        #print('output = ', convert_array_to_str(result[1]))
        score = model.score(test_data)
        print(score)
        models.append(model)
        with open("model_"+c[i]+str(j)+".pkl", "wb") as file: pickle.dump(model, file)


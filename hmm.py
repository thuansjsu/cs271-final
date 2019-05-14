import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split


from utils import read_find_and_convert_to_np_array, convert_array_to_str

model = hmm.MultinomialHMM(n_components=26, init_params="ste", n_iter=10, verbose=True)
#
data = np.array(read_find_and_convert_to_np_array('ciphers/column_transposition_cipher'))
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)
train = np.concatenate(train_data, axis=0)
length = [len(train_data[0]), len(train_data[1])]
#
model.fit(np.array([train]).T)

print('input = ', convert_array_to_str(test_data[0]))
result = model.decode(np.array([test_data[0]]).T, algorithm="viterbi")
print('output = ', convert_array_to_str(result[1]))


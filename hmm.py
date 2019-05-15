import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import read_file_and_convert_to_np_array, convert_array_to_str
from sklearn import preprocessing

ct_data = np.array(read_file_and_convert_to_np_array('ciphers/column_transposition_cipher'))
h_data = np.array(read_file_and_convert_to_np_array('ciphers/hill_cipher'))
pf_data = np.array(read_file_and_convert_to_np_array('ciphers/playfair_cipher', append_character='j'))
ss_data = np.array(read_file_and_convert_to_np_array('ciphers/simple_substitution_cipher'))
v_data = np.array(read_file_and_convert_to_np_array('ciphers/vigenere_cipher'))

result = []


class DataPoint:
    def __init__(self, cipher_text, label):
        self.cipher_text = cipher_text
        self.label = label
        self.score = [0, 0, 0, 0, 0]


data_points = []


def train_hmm(data, sequence_length, index):
    model = hmm.MultinomialHMM(n_components=26, init_params="ste", n_iter=2, verbose=True)
    train_data, test_data = train_test_split(data, test_size=0.05, shuffle=True)
    # train = np.concatenate(train_data, axis=0)
    # train = np.concatenate((train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]), axis=0)
    # length = [len(train_data[0]), len(train_data[1])]
    # model.fit(np.array([train]).T)
    if sequence_length != 0:
        train = np.concatenate(train_data, axis=0)[:sequence_length]
    else:
        train = np.concatenate(train_data, axis=0)
    model.fit(np.array([LabelEncoder().fit_transform(train)]).T)

    for text in test_data:
        # score = model.decode(np.array([text]).T)
        # result.append([score, index])
        # print(model.decode(np.array([test_data[0]]).T, algorithm="viterbi"))
        # score = model.score(np.array([text]).T)
        data_point = DataPoint(text, index)
        #     result.append(data_point)
        data_points.append(data_point)
    return model


# print('input = ', convert_array_to_str(test_data[0]))
# result = model.decode(np.array([test_data[0]]).T, algorithm="viterbi")
# print('output = ', convert_array_to_str(result[1]))

def train_and_score(dataset, sequence_length):
    # for index, data in enumerate([ct_data, h_data, pf_data, ss_data, v_data]):
    models = []
    # for index, data in enumerate([ct_data]):
    for index, data in enumerate(dataset):
        model = train_hmm(data, sequence_length, index)
        models.append(model)
    #
    for data_point in data_points:
        scores = [data_point.label]
        for index, model in enumerate(models):
            score = model.score(np.array([LabelEncoder().fit_transform(data_point.cipher_text)]).T)
            scores.append(score)
        result.append(scores)
    np.savetxt("abc.txt", result, fmt=["%d", "%s", "%s", "%s", "%s", "%s"], delimiter=', ')


train_and_score([ct_data, h_data, pf_data, ss_data, v_data], 10000)
dataset = np.loadtxt("abc.txt", delimiter=', ')
# X = dataset[:, 0].reshape(5, 100).T
X = dataset[:, 1:]
kmeans = KMeans(5, random_state=0)

model = kmeans.fit(X)
labels = model.predict(X)
print(labels)

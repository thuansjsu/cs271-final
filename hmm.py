import datetime

import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from utils import read_file_and_convert_to_np_array

ct_data = np.array(read_file_and_convert_to_np_array('ciphers/column_transposition_cipher'))
h_data = np.array(read_file_and_convert_to_np_array('ciphers/hill_cipher'))
pf_data = np.array(read_file_and_convert_to_np_array('ciphers/playfair_cipher', append_character='j'))
ss_data = np.array(read_file_and_convert_to_np_array('ciphers/simple_substitution_cipher'))
v_data = np.array(read_file_and_convert_to_np_array('ciphers/vigenere_cipher'))

result = []
data_point_sets = []
center_points = []


class DataPoint:
    def __init__(self, cipher_text, label):
        self.cipher_text = cipher_text
        self.label = label
        self.score = [0, 0, 0, 0, 0]


def train_hmm(data, sequence_length, n_iter, test_size, index):
    model = hmm.MultinomialHMM(n_components=26, init_params="ste", n_iter=n_iter, verbose=True)
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=True)
    if sequence_length != 0:
        train = np.concatenate(train_data, axis=0)[:sequence_length]
    else:
        train = np.concatenate(train_data, axis=0)
    model.fit(np.array([train]).T)

    test_result = []
    for text in test_data:
        data_point = DataPoint(text, index)
        test_result.append(data_point)
    return model, test_result


def train_and_score(dataset, sequence_length, n_iter, test_size):
    models = []
    # train 5 models
    for index, data in enumerate(dataset):
        model, test_result = train_hmm(data, sequence_length, n_iter, test_size, index)
        models.append(model)
        data_point_sets.append(test_result)

    # score the sample
    for data_points in data_point_sets:
        for ind, data_point in enumerate(data_points):
            scores = [data_point.label]
            center_point = []
            for index, model in enumerate(models):
                score = model.score(np.array([data_point.cipher_text]).T)
                scores.append(score)
                center_point.append(score)
            result.append(scores)
            if ind == 0:
                center_points.append(center_point)
    file_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%m") + '.txt'
    np.savetxt(file_name, result, fmt=["%d", "%10.2f", "%10.2f", "%10.2f", "%10.2f", "%10.2f"], delimiter=', ')
    return file_name


test_size = 0.05
file_name = train_and_score([ct_data, h_data, pf_data, ss_data, v_data], 0, 50, test_size)
# samples = np.loadtxt("2019-05-15 18:40:05.txt", delimiter=', ')
samples = np.loadtxt(file_name, delimiter=', ')
X = samples[:, 1:]
init_centers = np.array(center_points)
kmeans = KMeans(5, init=init_centers,  max_iter=1000)

kmeans_model = kmeans.fit(X)
labels = kmeans_model.predict(X)
print(labels)

number_of_text = int(test_size * 1000)
for i in range(0, 5):
    correct_count = 0
    for j in range(0, number_of_text * (i + 1)):
        if int(labels[j]) == i:
            correct_count += 1
    print("accuracy for {} is: {}".format(i, correct_count / number_of_text))
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


class DataPoint:
    def __init__(self, cipher_text, label):
        self.cipher_text = cipher_text
        self.label = label
        self.score = [0, 0, 0, 0, 0]


class HmmCluster:
    def __init__(self, sequence_length, test_size, n_iter):
        self.result = []
        self.data_point_sets = []
        self.center_points = []
        self.test_size = test_size
        self.sequence_length = sequence_length
        self.n_iter = n_iter

    def train_hmm(self, data, index):
        model = hmm.MultinomialHMM(n_components=26, init_params="ste", n_iter=self.n_iter, verbose=True)
        train_data, test_data = train_test_split(data, test_size=self.test_size, shuffle=True)
        if self.sequence_length != 0:
            train = np.concatenate(train_data, axis=0)[:self.sequence_length]
        else:
            train = np.concatenate(train_data, axis=0)
        model.fit(np.array([train]).T)

        test_result = []
        for text in test_data:
            data_point = DataPoint(text, index)
            test_result.append(data_point)
        return model, test_result

    def train_and_score(self, dataset):
        models = []
        # train 5 models
        for index, data in enumerate(dataset):
            model, test_result = self.train_hmm(data, index)
            models.append(model)
            self.data_point_sets.append(test_result)

        # score the sample
        for data_points in self.data_point_sets:
            for ind, data_point in enumerate(data_points):
                scores = [data_point.label]
                center_point = []
                for index, model in enumerate(models):
                    score = model.score(np.array([data_point.cipher_text]).T)
                    scores.append(score)
                    center_point.append(score)
                self.result.append(scores)
                if ind == 0:
                    self.center_points.append(center_point)
        file_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%m") + '.txt'
        np.savetxt(file_name, self.result, fmt=["%d", "%10.2f", "%10.2f", "%10.2f", "%10.2f", "%10.2f"], delimiter=', ')
        return file_name


results = []
for i in [5, 10, 15, 20, 25, 30]:
    for n_iter in [1, 2, 5, 10, 15, 20]:
        sequence_length = i * 1000
        hmm_cluster = HmmCluster(sequence_length, 0.05, n_iter)
        file_name = hmm_cluster.train_and_score([ct_data, h_data, pf_data, ss_data, v_data])
        samples = np.loadtxt(file_name, delimiter=', ')
        X = samples[:, 1:]
        init_centers = np.array(hmm_cluster.center_points)
        kmeans = KMeans(5, init=init_centers, max_iter=1000)
        kmeans_model = kmeans.fit(X)
        labels = kmeans_model.predict(X)

        test_size = 0.05
        number_of_text = int(test_size * 1000)
        result = [sequence_length, n_iter]
        for n in range(0, 5):
            correct_count = 0
            for j in range(0, number_of_text * (n + 1)):
                if int(labels[j]) == n:
                    correct_count += 1
            print("accuracy for {} is: {}".format(i, correct_count / number_of_text))
            result.append(correct_count / number_of_text)
        results.append(result)
print(results)
np.savetxt('result.txt', results, fmt=["%d", "%d", "%10.2f", "%10.2f", "%10.2f", "%10.2f", "%10.2f"], delimiter=', ')

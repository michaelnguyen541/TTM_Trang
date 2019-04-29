import os
import sys
import numpy as np
import math

class Preprocess(object):
    def __init__(self):
        return

    def fit(self, input_folder_path, filename):  # ex: filename = 'AlarmClock'
        self.filename = filename + '.txt'  # ex: self.filename = 'AlarmClock.txt'
        input_folder_path_2 = os.path.join(input_folder_path, '../data_khachsan')
        if (not os.path.exists(input_folder_path_2)):
            os.mkdir(input_folder_path_2)
        self.input_file_path = os.path.join(input_folder_path, self.filename)
        self.vocab_file_path = os.path.join(input_folder_path_2, filename + '.vocab')
        self.docs_file_path = os.path.join(input_folder_path_2, filename + '.docs')
        self.tf_file_path = os.path.join(input_folder_path_2, filename + '.tf')
        self.tfidf_file_path = os.path.join(input_folder_path_2, filename + '.tfidf')

        with open(self.input_file_path, 'r', encoding='utf8') as fin: # i deleted the phrase: encoding='utf8'
            self.X = fin.readlines()

        for i in range(len(self.X)):
            self.X[i] = self.X[i].lower()

        self.build_dictionary()
        return

    def build_dictionary(self):
        self.dict = {}

        feature = 0
        for line in self.X:
            split_words = line.split()
            for word in split_words:
                # word = word.lower()
                if self.dict.get(word) is None:
                    self.dict[word] = feature
                    feature += 1

        self.writeDict(self.dict, self.vocab_file_path)
        return self

    def writeDict(self, dict, filename, sep=':'):
        text_file = open(filename, 'w', encoding='utf8') # i deleted the phrase: encoding='utf8'
        for word in dict:
            line = str(dict[word]) + sep + word
            text_file.write(line)
            text_file.write('\n')
        text_file.close()

    def transform(self, save_tf):
        # Load dictionary
        if self.dict is None:
            self.dict = self.readDict()

        newlines = self.bn_vectorizer(self.X)
        text_file = open(self.docs_file_path, 'w') # i deleted the phrase: encoding='utf8'
        for line in newlines:
            text_file.write(line)
            text_file.write('\n')
        text_file.close()
        return self.tf_and_tfidf_vectorizer(self.X, save_tf)

    def readDict(self, filename, sep=':'):
        dict = {}
        with open(filename, 'r') as fin: # i deleted the phrase: encoding='utf8'
            lines = fin.readlines()
            for line in lines:
                elem = line.split(sep)
                dict[elem[1]] = int(elem[0])
        return dict

    def bn_vectorizer(self, X):
        lines = []
        matrix = np.zeros((len(X), len(self.dict)), dtype=float)
        for i in range(len(X)):
            line = ''
            split_words = X[i].split()
            for word in split_words:
                # word = word.lower()
                j = self.dict.get(word)
                line += str(j) + ' '
            line = line[0:len(line) - 1]
            lines.append(line)
        return lines

    def tf_and_tfidf_vectorizer(self, X, save_tf):
        tf_matrix = np.zeros((len(X), len(self.dict)), dtype=float)
        tfidf_matrix = np.zeros((len(X), len(self.dict)), dtype=float)
        for i in range(len(X)):
            split_words = X[i].split()
            for word in split_words:
                # word = word.lower()
                tf = self.compute_tf(word, X[i])
                idf = self.compute_idf(word, X)
                j = int(self.dict.get(word))
                tf_matrix[i, j] = tf
                tfidf_matrix[i, j] = tf * idf

        # Write tf matrix and td.idf matrix
        if (save_tf):
            np.savetxt(self.tf_file_path, tf_matrix)
            np.savetxt(self.tfidf_file_path, tfidf_matrix)

        return (tf_matrix, tfidf_matrix)

    def compute_tf(self, word, sentence):
        split_words = sentence.split()
        word_freq = split_words.count(word)
        tf = word_freq / len(split_words)
        return tf

    def compute_idf(self, word, X):
        n_docs_with_word = 0
        for i in range(len(X)):
            if word in X[i]:
                n_docs_with_word += 1

        n_docs = len(X)
        idf = n_docs / n_docs_with_word
        return math.log(idf)

    # def get_tf_and_tfidf_matrix(filename):
    #     prc = Preprocess()
    #     prc.fit(filename)
    #     return prc.transform()

if __name__ == '__main__':
    # Chuyen file text sang dang .vocab .docs
	parent_1_paths = ['../../data/input/origin']

	for i in range(len(parent_1_paths)):
		path = parent_1_paths[i]
		folder_names = ['data_ks']
		for folder_name in folder_names:
			print(path, end='/')
			print(folder_name, end='.txt\n')
			prc = Preprocess()
			prc.fit(path, folder_name)
			save_tf = True
			tmp = prc.transform(save_tf)

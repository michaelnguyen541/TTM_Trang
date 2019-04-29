import os
import sys
import numpy as np
from sklearn import tree
import csv
sys.path.insert(0, '../util')
import numpy as np
from word_list_from_hidden_topics import get_word_list_from_hidden_topics

#- Biểu diễn từng câu trong 1493 câu qua (n * m) từ ở trên
#  + Biểu diễn bằng binary: Với mỗi văn bản trong 1493 văn bản, biểu diễn nó thành một vector (n * m) chiều.
#    Chiều thứ i nhận giá trị 1/0 tương ứng với từ thứ i trong danh sách (n * m) từ có nằm trong văn bản đang
#    xét hay không. Như vậy, ta sẽ có ma trận có số hàng là số văn bản, số cột là (n * m).

if (__name__=='__main__'):
    # Doc word list from hidden topics
    word_list = get_word_list_from_hidden_topics()
    # print(word_list)

    # Doc tf va tfidf cua 1493 du lieu
    # data_folder_path = '../../data/input/data_khachsan'
    # tf_path = os.path.join(data_folder_path, 'data_ks.tf')
    # tfidf_path = os.path.join(data_folder_path, 'data_ks.tfidf')
    # tf = np.loadtxt(tf_path, dtype='float')
    # tfidf= np.loadtxt(tfidf_path, dtype='float')

    # Doc 1493 van ban
    data_folder_path = '../../data/input/origin'
    data_path = os.path.join(data_folder_path, 'data_ks.txt')
    with open(data_path, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, delimiter=' ')
        binary = [] # Bieu dien du lieu nhi phan (binary), sau nay chung ta co the bieu dien theo tf hoac tfidf
        data = []
        for row in reader:
            tmp = []
            for word in word_list:
                if (word in row):
                    tmp.append(1)
                else:
                    tmp.append(0)
            binary.append(tmp)
        binary = np.asarray(binary)
        # print(binary[1])
        # print(binary.shape)

        # x = np.sum(binary, axis=1)
        # print(x)
        # print(len(x))
        # print(max(x))
        # print(min(x))

    # Doc tap nhan
    labels_path = os.path.join(data_folder_path, 'labels.csv')
    labels = np.loadtxt(labels_path, dtype='int')
    Y = abs(labels)

    # Classify and predict
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(binary[:1200, :], Y[:1200, :]) # Chia 1493 thanh 2 tap: [1] gom 1200 training docs, [2] gom 293 test docs
    tmp = clf.predict(binary[1200:, :])
    print(tmp)
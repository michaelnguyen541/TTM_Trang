import os
import numpy as np
import csv

# - Với mỗi khía cạnh, tìm n chủ đề ẩn (n = 6). Mỗi chủ đề ẩn được biểu diễn qua 1 danh sách các từ liên quan
# đến chủ đề ẩn đó, số từ của mỗi chủ đề ẩn có thể khác nhau.
# - Với mỗi chủ đề ẩn, mình kỳ vọng lấy ra m từ (m = 30). Nếu ở một chủ đề ẩn nhiều hơn hoặc bằng 30 từ thì
# chỉ lấy đến 30 từ, còn nếu nhỏ hơn 30 từ thì lấy từ chủ đề ẩn khác sang, để đảm bảo sự bằng nhau trong dữ liệu
# của các hàng. Các từ lấy từ chủ đề ẩn ra sẽ được ghép với nhau thành một danh sách (list) gồm n * m từ
# (6 * 30 = 180 từ).

def get_word_list_from_hidden_topics():
    n_wiht = 30 # viet tat cua: n_words_in_hidden_topics

    with open('../../data/input/origin/hotelTTM-target-nhân_viên-PT-6-1-Iter-250-svn-1.0-stn-1.0targeted_topic_words.csv', 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        hidden_topics = [] # kieu list
        for row in reader:
            break
        for row in reader:
            hidden_topics.append(row)

    # chung ta muon hidden_topics la mot numpy array (2D), nen se chuyen tu list of lists sang numpy array (2D)
    hidden_topics = np.asarray(hidden_topics) # la 1 ham (function) cua numpy
    hidden_topics = hidden_topics[:, :-1]

    lengths = []
    m, n = hidden_topics.shape
    print(hidden_topics.shape)
    for i in range(n):
        ok = 0
        for j in range(m):
            if (hidden_topics[j][i] == ''):
                print(j, end=' ')
                ok = 1
                break
        if (ok == 0):
            print(m)

    res = []
    for i in range(n):
        for j in range(n_wiht):
            if (hidden_topics[j][i] == ''):
                print('You have a problem!')
            res.append(hidden_topics[j][i])

    return res
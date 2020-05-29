'''
Created on May 29, 2020
@author: nakaizura
'''

import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie': 'ratings.dat', 'book': 'BX-Book-Ratings.csv', 'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'news': 0})


#这个py的目的是从原数据集中得到需要的字段，然后对齐item和实体的id（表示了同一个物品）
#按照对齐的id，对推荐rating和图谱kg都制作一个方便处理final文件
#目标：
#kg_final.txt：h，r，t
#rating_final.txt：userid，itemid，rating


#将item和实体entity给对应起来
def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()#用户的正例
    user_neg_ratings = dict()#用户的负例

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])#按分隔符分割

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))#移除最前和最后的无关标记

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  #如果该item不在最后的数据集中
            continue
        item_index = item_index_old2new[item_index_old]#得到转换后的index

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:#大于THRESHOLD被认为是有评分，为正例
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:#不然就加入到负例的列表中
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    #对应好之后，按格式写入一个final文件便于后面的模型处理
    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:#正例
            writer.write('%d\t%d\t1\n' % (user_index, item))#user_index, item，1
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:#负例
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))#user_index, item，0
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    #目标也同样是制作一个final文件便于模型的处理
    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')

    files = []
    if DATASET == 'movie':
        files.append(open('../data/' + DATASET + '/kg_part1_rehashed.txt', encoding='utf-8'))
        files.append(open('../data/' + DATASET + '/kg_part2_rehashed.txt', encoding='utf-8'))
    else:
        files.append(open('../data/' + DATASET + '/kg_rehashed.txt', encoding='utf-8'))

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            #得到三元组
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            #转换成统一的id
            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            #写入
            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)#可复现随机种子

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset

    entity_id2index = dict()#实体的转换
    relation_id2index = dict()#关系的转换
    item_index_old2new = dict()#item的转换

    read_item_index_to_entity_id_file()#将item和实体id对齐
    convert_rating()#转换成统一index，写final文件
    convert_kg()#转换成统一的index，写final文件

    print('done')

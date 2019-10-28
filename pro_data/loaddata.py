'''
Data pre process

@author:
Chong Chen (cstchenc@163.com)

@ created:
25/8/2017
@references:
'''
import os
import json
import pandas as pd
import pickle
import numpy as np
TPS_DIR = '../data/music'
TP_file = os.path.join(TPS_DIR, 'amazon_instant_video_train.json')
TP_file_test = os.path.join(TPS_DIR, 'amazon_instant_video_test.json' )
TP_file_valid = os.path.join(TPS_DIR, 'amazon_instant_video_dev.json')

f = open(TP_file)
f_test = open(TP_file_test)
f_valid = open(TP_file_valid)
np.random.seed(2017)

def read(f):
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    for line in f:
        js = json.loads(line)
        if str(js['reviewerID']) == 'unknown':
            print "unknown"
            continue
        if str(js['asin']) == 'unknown':
            print "unknown2"
            continue
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID']) + ',')
        items_id.append(str(js['asin']) + ',')
        ratings.append(str(js['overall']))
    data = pd.DataFrame({'user_id': pd.Series(users_id),
                         'item_id': pd.Series(items_id),
                         'ratings': pd.Series(ratings),
                         'reviews': pd.Series(reviews)})[['user_id', 'item_id', 'ratings', 'reviews']]
    return reviews,users_id,items_id,ratings,data


reviews_train, users_id_train, items_id_train, ratings_train, data_train = read(f)
reviews_test, users_id_test, items_id_test, ratings_test, data_test = read(f_test)
reviews_valid, users_id_valid, items_id_valid, ratings_valid, data_valid = read(f_valid)


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
usercount_train, itemcount_train = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
usercount_test, itemcount_test = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
usercount_valid, itemcount_valid = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
unique_uid_train = usercount_train.index
unique_sid_train = itemcount_train.index
unique_uid_test = usercount_test.index
unique_sid_test = itemcount_test.index
unique_uid_valid = usercount_valid.index
unique_sid_valid = itemcount_valid.index
item2id_train = dict((sid, i) for (i, sid) in enumerate(unique_sid_train))
user2id_train = dict((uid, i) for (i, uid) in enumerate(unique_uid_train))
item2id_test = dict((sid, i) for (i, sid) in enumerate(unique_sid_test))
user2id_test = dict((uid, i) for (i, uid) in enumerate(unique_uid_test))
item2id_valid = dict((sid, i) for (i, sid) in enumerate(unique_sid_valid))
user2id_valid = dict((uid, i) for (i, uid) in enumerate(unique_uid_valid))
def numerize(tp,user2id,item2id):
    uid = map(lambda x: user2id[x], tp['user_id'])
    sid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp

data_train=numerize(data_train,user2id_train,item2id_train)
data_test=numerize(data_test,user2id_test,item2id_test)
data_valid=numerize(data_valid,user2id_valid,item2id_valid)

tp_rating_train=data_train[['user_id','item_id','ratings']]
tp_rating_test=data_test[['user_id','item_id','ratings']]
tp_rating_valid=data_valid[['user_id','item_id','ratings']]



n_ratings_train = tp_rating_train.shape[0]
n_ratings_test = tp_rating_test.shape[0]
n_ratings_valid = tp_rating_valid.shape[0]
#test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
#test_idx = np.zeros(n_ratings, dtype=bool)
#test_idx[test] = True

#tp_1 = tp_rating[test_idx]
#tp_train= tp_rating[~test_idx]

#data2=data[test_idx]
#data=data[~test_idx]


#n_ratings = tp_1.shape[0]
#test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

#test_idx = np.zeros(n_ratings, dtype=bool)
#test_idx[test] = True

#tp_test = tp_1[test_idx]
#tp_valid = tp_1[~test_idx]
tp_rating_train.to_csv(os.path.join(TPS_DIR, 'train.csv'), index=False,header=None)
tp_rating_valid.to_csv(os.path.join(TPS_DIR, 'valid.csv'), index=False,header=None)
tp_rating_test.to_csv(os.path.join(TPS_DIR, 'test.csv'), index=False,header=None)

user_reviews={}
item_reviews={}
user_rid={}
item_rid={}
for i in data_train.values:
    if user_reviews.has_key(i[0]):
        user_reviews[i[0]].append(i[3])
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]] = [i[1]]
        user_reviews[i[0]] = [i[3]]
    if item_reviews.has_key(i[1]):
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]] = [i[0]]


for i in data_test.values:
    if user_reviews.has_key(i[0]):
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=['0']
    if item_reviews.has_key(i[1]):
        l=1
    else:
        item_reviews[i[1]] = [0]
        item_rid[i[1]]=['0']



pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

'''
NARRE
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:

'''

import numpy as np
import tensorflow as tf
import pickle
import datetime
import NARRE

# tf.flags.DEFINE_string("dir","../data/music/", "Directory")
tf.flags.DEFINE_string("word2vec", "../data/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("valid_data", "../data/music/data.valid", " Data for validation")
tf.flags.DEFINE_string("para_data", "../data/music/data.para", "Data parameters")
tf.flags.DEFINE_string("train_data", "../data/music/data.train", "Data for training")
tf.flags.DEFINE_string("test_data", "../data/music/data.test", " Data for testing")
tf.flags.DEFINE_string("model_path", "../model", "Model path")
# ==================================================

# Model Hyperparameters
# tf.flags.DEFINE_string("word2vec", "./data/rt-polaritydata/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 0.8,

        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae, u_a, i_a, fm = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.u_a, deep.i_a, deep.score],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, accuracy * accuracy, mae, u_a, i_a, fm


def dev_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    # import sys
    # FLAGS(sys.argv)
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading data...")
    pkl_file = open(FLAGS.para_data, 'rb')

    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    review_len_u = para['review_len_u']
    review_len_i = para['review_len_i']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    valid_length = para['valid_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']

    np.random.seed(2017)
    random_seed = 2017
    print user_num
    print item_num
    print review_num_u
    print review_len_u
    print review_num_i
    print review_len_i
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            deep = NARRE.NARRE(
                review_num_u=review_num_u,
                review_num_i=review_num_i,
                review_len_u=review_len_u,
                review_len_i=review_len_i,
                user_num=user_num,
                item_num=item_num,
                num_classes=1,
                user_vocab_size=len(vocabulary_user),
                item_vocab_size=len(vocabulary_item),
                embedding_size=FLAGS.embedding_dim,
                embedding_id=32,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                attention_size=32,
                n_latent=32)
            tf.set_random_seed(random_seed)
            print user_num
            print item_num
            global_step = tf.Variable(0, name="global_step", trainable=False)


            # optimizer = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=1e-8).minimize(deep.loss)
            optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)

            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)



            sess.run(tf.initialize_all_variables())


            if FLAGS.word2vec:
                # initial matrix with random uniform
                u = 0
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec u file {}\n".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = 0

                        if word in vocabulary_user:
                            u = u + 1
                            idx = vocabulary_user[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(deep.W1.assign(initW))
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec i file {}\n".format(FLAGS.word2vec))

                item = 0
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = 0
                        if word in vocabulary_item:
                            item = item + 1
                            idx = vocabulary_item[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)

                sess.run(deep.W2.assign(initW))
                print item

            saver = tf.train.Saver()

            epoch = 1
            best_mae = 5
            best_rmse = 5
            best_mse = 25
            train_mae = 0
            train_rmse = 0
            train_mse = 0
            best_rmse_valid = 100
            best_mse_valid = 10000
            best_mae_valid = 100

            valid_rmse_scores, valid_mae_scores, valid_mse_scores = [], [], []
            test_rmse_scores, test_mae_scores, test_mse_scores = [], [], []

            pkl_file = open(FLAGS.train_data, 'rb')
            train_data = pickle.load(pkl_file)
            train_data = np.array(train_data)
            pkl_file.close()

            pkl_file = open(FLAGS.valid_data, 'rb')
            valid_data = pickle.load(pkl_file)
            valid_data = np.array(valid_data)
            pkl_file.close()

            pkl_file = open(FLAGS.test_data, 'rb')
            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            pkl_file.close()

            data_size_train = len(train_data)
            data_size_valid = len(valid_data)
            data_size_test = len(test_data)
            batch_size = FLAGS.batch_size

            ll = int(len(train_data) / batch_size)
            for epoch in range(FLAGS.num_epochs):
                # Shuffle the data at each epoch
                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                for batch_num in range(ll):

                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]

                    uid, iid, reuid, reiid, y_batch = zip(*data_train)
                    u_batch = []
                    i_batch = []
                    for i in range(len(uid)):
                        u_batch.append(u_text[uid[i][0]])
                        i_batch.append(i_text[iid[i][0]])
                    u_batch = np.array(u_batch)
                    i_batch = np.array(i_batch)

                    t_rmse, t_mse, t_mae, u_a, i_a, fm = train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch,
                                                                    batch_num)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mse += t_mse
                    train_mae += t_mae
                    if batch_num % 500 == 0 and batch_num > 1:
                        print("\nEvaluation:")
                        print batch_num

                        loss_s = 0
                        accuracy_s = 0
                        mae_s = 0

                        ll_valid = int(len(valid_data) / batch_size) + 1
                        for batch_num in range(ll_valid):
                            start_index = batch_num * batch_size
                            end_index = min((batch_num + 1) * batch_size, data_size_valid)
                            data_valid = valid_data[start_index:end_index]

                            userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_valid)
                            u_valid = []
                            i_valid = []
                            for i in range(len(userid_valid)):
                                u_valid.append(u_text[userid_valid[i][0]])
                                i_valid.append(i_text[itemid_valid[i][0]])
                            u_valid = np.array(u_valid)
                            i_valid = np.array(i_valid)

                            loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid,
                                                           y_valid)
                            loss_s = loss_s + len(u_valid) * loss
                            accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                            mae_s = mae_s + len(u_valid) * mae
                        print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}, mse_valid {:g}".format(
                            loss_s / valid_length, np.sqrt(accuracy_s / valid_length), mae_s / valid_length,
                            accuracy_s / valid_length))
                        rmse = np.sqrt(accuracy_s / valid_length)
                        mse = accuracy_s / valid_length
                        mae = mae_s / valid_length
                        if best_rmse > rmse:
                            best_rmse = rmse
                        if best_mae > mae:
                            best_mae = mae
                        if best_mse > mse:
                            best_mse = mse
                        print("")

                print str(epoch) + ':\n'
                print("\nEvaluation:")
                print "train:rmse,mae,mse:", train_rmse / ll, train_mae / ll, train_mse / ll
                u_a = np.reshape(u_a[0], (1, -1))
                i_a = np.reshape(i_a[0], (1, -1))

                print u_a
                print i_a
                train_rmse = 0
                train_mae = 0
                train_mse = 0

                loss_s = 0
                accuracy_s_valid = 0
                mae_s_valid = 0

                ll_valid = int(len(valid_data) / batch_size) + 1
                for batch_num in range(ll_valid):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_valid)
                    data_valid = valid_data[start_index:end_index]

                    userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_valid)
                    u_valid = []
                    i_valid = []
                    for i in range(len(userid_valid)):
                        u_valid.append(u_text[userid_valid[i][0]])
                        i_valid.append(i_text[itemid_valid[i][0]])
                    u_valid = np.array(u_valid)
                    i_valid = np.array(i_valid)

                    loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid, y_valid)
                    loss_s = loss_s + len(u_valid) * loss
                    accuracy_s_valid = accuracy_s_valid + len(u_valid) * np.square(accuracy)
                    mae_s_valid = mae_s_valid + len(u_valid) * mae
                print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}, mse_valid {:g}".format(loss_s / valid_length,
                                                                                 np.sqrt(accuracy_s_valid / valid_length),
                                                                                 mae_s_valid / valid_length,
                                                                                 accuracy_s_valid / valid_length))
                rmse_valid = np.sqrt(accuracy_s_valid / valid_length)
                mse_valid = accuracy_s_valid / valid_length
                mae_s_valid = mae_s_valid / valid_length
                if best_rmse_valid > rmse_valid:
                    best_rmse_valid = rmse_valid
                    save_path = saver.save(sess, "%s/model_%s.ckpt" % (FLAGS.model_path, str(epoch)))
                    best_epoch = epoch
                if best_mae_valid > mae_s_valid:
                    best_mae_valid = mae_s_valid
                if best_mse_valid > mse_valid:
                    best_mse_valid = mse_valid

                loss_s = 0
                accuracy_s_test = 0
                mae_s_test = 0

                ll_test = int(len(test_data) / batch_size) + 1
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index:end_index]

                    userid_test, itemid_test, reuid, reiid, y_test = zip(*data_test)

                    # print("REUID test", reuid[0].shape)
                    u_test = []
                    i_test = []
                    for i in range(len(userid_test)):
                        u_test.append(u_text[userid_test[i][0]])
                        i_test.append(i_text[itemid_test[i][0]])
                    u_test = np.array(u_test)
                    i_test = np.array(i_test)

                    loss, accuracy, mae = dev_step(u_test, i_test, userid_test, itemid_test, reuid, reiid,
                                                         y_test)
                    loss_s += len(u_test) * loss
                    accuracy_s_test += len(u_test) * np.square(accuracy)
                    mae_s_test += len(u_test) * mae
                print ("loss_test {:g}, rmse_test {:g}, mae_test {:g}, mse_test {:g}".format(loss_s / test_length,
                                                                                    np.sqrt(
                                                                                        accuracy_s_test / test_length),
                                                                                    mae_s_test / test_length,
                                                                                             accuracy_s_test / test_length))

                valid_rmse = np.sqrt(accuracy_s_valid / valid_length)
                valid_mae = mae_s_valid / valid_length
                valid_mse = accuracy_s_valid / valid_length

                test_rmse = np.sqrt(accuracy_s_test / test_length)
                test_mae = mae_s_test / test_length
                test_mse = accuracy_s_test / test_length

                valid_rmse_scores.append(valid_rmse)
                valid_mae_scores.append(valid_mae)
                valid_mse_scores.append(valid_mse)

                test_rmse_scores.append(test_rmse)
                test_mae_scores.append(test_mae)
                test_mse_scores.append(test_mse)

            index = valid_rmse_scores.index(min(valid_rmse_scores))

            print "Best valid: RMSE {:g}, MAE {:g}, MSE {:g}".format(valid_rmse_scores[index], valid_mae_scores[index],
                                                           valid_mse_scores[index])

            saver.restore(sess, "%s/model_%s.ckpt" % (FLAGS.model_path, str(index)))

            y_true = []
            y_pred = []
            loss_s = 0
            accuracy_s_test = 0
            mae_s_test = 0

            ll_test = int(len(test_data) / batch_size) + 1
            for batch_num in range(ll_test):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size_test)
                data_test = test_data[start_index:end_index]

                userid_test, itemid_test, reuid, reiid, y_test = zip(*data_test)
                u_test = []
                i_test = []
                for i in range(len(userid_test)):
                    u_test.append(u_text[userid_test[i][0]])
                    i_test.append(i_text[itemid_test[i][0]])
                u_test = np.array(u_test)
                i_test = np.array(i_test)
                loss, accuracy, mae = dev_step(u_test, i_test, userid_test, itemid_test, reuid, reiid, y_test)
                loss_s += len(u_test) * loss
                accuracy_s_test += len(u_test) * np.square(accuracy)
                mae_s_test += len(u_test) * mae
            print ("loss_test {:g}, rmse_test {:g}, mae_test {:g}, mse_test {:g}".format(loss_s / test_length,
                                                                                             np.sqrt(
                                                                                                 accuracy_s_test / test_length),
                                                                                             mae_s_test / test_length,
                                                                                             accuracy_s_test / test_length))




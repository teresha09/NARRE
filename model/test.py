import numpy as np
import tensorflow as tf
import pickle
import datetime
import NARRE

tf.flags.DEFINE_string("word2vec", "../data/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("test_data", "../data/music/data.test", " Data for test")
tf.flags.DEFINE_string("para_data", "../data/music/data.para", "Data parameters")

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


def test_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a test set

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
        [global_step, deep.loss, deep.accuracy, deep.mae], feed_dict)
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
    test_length = para['valid_length']
    u_text = para['u_text']
    i_text = para['i_text']

    np.random.seed(2017)
    random_seed = 2017
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("/home/egor/PycharmProjects/NARRE/model/model.meta")
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        saver.restore(sess, './model')
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
        global_step = tf.Variable(sess.run('global_step:0'))
        optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
        train_op = optimizer

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

        best_mae = 5
        best_rmse = 5
        best_mse = 25
        train_mae = 0
        train_rmse = 0
        train_mse = 0

        pkl_file = open(FLAGS.test_data, 'rb')
        test_data = pickle.load(pkl_file)
        test_data = np.array(test_data)
        pkl_file.close()
        data_size_test = len(test_data)
        batch_size = FLAGS.batch_size
        loss_s = 0
        accuracy_s = 0
        mae_s = 0
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
            loss, accuracy, mae = test_step(u_test, i_test, userid_test, itemid_test, reuid, reiid,
                                            y_test)
            loss_s = loss_s + len(u_test) * loss
            accuracy_s = accuracy_s + len(u_test) * np.square(accuracy)
            mae_s = mae_s + len(u_test) * mae
        print ("loss_test {:g}, rmse_test {:g}, mae_test {:g}".format(loss_s / test_length,
                                                                      np.sqrt(accuracy_s / test_length),
                                                                      mae_s / test_length,
                                                                      accuracy_s / test_length))
        rmse = np.sqrt(accuracy_s / test_length)
        mse = accuracy_s / test_length
        mae = mae_s / test_length
        print 'rmse:', rmse
        print 'mae:', mae
        print 'mse:', mse

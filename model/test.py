import numpy as np
import tensorflow as tf
import pickle
import datetime
import NARRE

# tf.flags.DEFINE_string("dir","../data/music/", "Directory")
tf.flags.DEFINE_string("word2vec", "../data/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("valid_data", "../data/music/data.test", " Data for validation")
tf.flags.DEFINE_string("test_data", "../data/music/data.test", " Data for validation")
tf.flags.DEFINE_string("para_data", "../data/music/data.para", "Data parameters")
tf.flags.DEFINE_string("train_data", "../data/music/data.train", "Data for training")
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
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("model-0.meta")
    for tensor in tf.get_default_graph().get_operations():
        print (tensor.name)

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
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        iidW = sess.graph.get_tensor_by_name("iidW:0")
        uidW = sess.graph.get_tensor_by_name("uidW:0")
        W1 = sess.graph.get_tensor_by_name("user_embedding/W1:0")
        W2 = sess.graph.get_tensor_by_name("item_embedding/W2:0")
        W = sess.graph.get_tensor_by_name("user_conv-maxpool-3/W:0")
        b = sess.graph.get_tensor_by_name("user_conv-maxpool-3/b:0")
        w = sess.graph.get_tensor_by_name("item_conv-maxpool-3/W:0")
        B = sess.graph.get_tensor_by_name("item_conv-maxpool-3/b:0")
        Wau = sess.graph.get_tensor_by_name("attention/Wau:0")
        Wru = sess.graph.get_tensor_by_name("attention/Wru:0")
        Wpu = sess.graph.get_tensor_by_name("attention/Wpu:0")
        bau = sess.graph.get_tensor_by_name("attention/bau:0")
        bbu = sess.graph.get_tensor_by_name("attention/bbu:0")
        Wai = sess.graph.get_tensor_by_name("attention/Wai:0")
        Wri = sess.graph.get_tensor_by_name("attention/Wri:0")
        Wpi = sess.graph.get_tensor_by_name("attention/Wpi:0")
        bai = sess.graph.get_tensor_by_name("attention/bai:0")
        bbi = sess.graph.get_tensor_by_name("attention/bbi:0")
        iidmf = sess.graph.get_tensor_by_name("get_fea/iidmf:0")
        uidmf = sess.graph.get_tensor_by_name("get_fea/uidmf:0")
        Wu = sess.graph.get_tensor_by_name("get_fea/Wu:0")
        bu = sess.graph.get_tensor_by_name("get_fea/bu:0")
        Wi = sess.graph.get_tensor_by_name("get_fea/Wi:0")
        bi = sess.graph.get_tensor_by_name("get_fea/bi:0")
        Wmul = sess.graph.get_tensor_by_name("ncf/wmul:0")
        uidW2 = sess.graph.get_tensor_by_name("ncf/uidW2:0")
        iidW2 = sess.graph.get_tensor_by_name("ncf/iidW2:0")
        bised = sess.graph.get_tensor_by_name("ncf/bias:0")
        global_step = sess.graph.get_tensor_by_name('global_step:0')

        tf.set_random_seed(random_seed)
        print user_num
        print item_num

        sess.run(tf.initialize_all_variables())


        best_mae = 5
        best_rmse = 5
        best_mse = 25

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

            userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
            u_valid = []
            i_valid = []
            for i in range(len(userid_valid)):
                u_valid.append(u_text[userid_valid[i][0]])
                i_valid.append(i_text[itemid_valid[i][0]])
            u_valid = np.array(u_valid)
            i_valid = np.array(i_valid)

            loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid, y_valid)
            loss_s = loss_s + len(u_valid) * loss
            accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
            mae_s = mae_s + len(u_valid) * mae
        print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}, mse_valid {:g}".format(loss_s / test_length,
                                                                                         np.sqrt(
                                                                                             accuracy_s / test_length),
                                                                                         mae_s / test_length,
                                                                                         accuracy_s / test_length))
        rmse = np.sqrt(accuracy_s / test_length)
        mse = accuracy_s / test_length
        mae = mae_s / test_length
        if best_rmse > rmse:
            best_rmse = rmse
        if best_mae > mae:
            best_mae = mae
        if best_mse > mse:
            best_mse = mse
        print ""
        print best_rmse
        print best_mae
        print best_mse

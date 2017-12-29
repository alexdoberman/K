# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import config
from data_load import *
from data_generator import *



def predict_by_score_1(current_prediction):
    predict_class_id        = np.argmax(current_prediction, axis=1)[0]
    return predict_class_id

def predict_by_score_2(current_prediction):

    alfa = 0.98

    score_with_unknown    = np.max(current_prediction, axis=1)[0]
    score_without_unknown = np.max(current_prediction[:,0:11], axis=1)[0]
    score_unknown         = current_prediction[0,11]

    div_score = score_without_unknown/(score_with_unknown + 0.00001)

    predict_class_id = -1

    if div_score < alfa:
        predict_class_id        = 11
    else:
        predict_class_id        = np.argmax(current_prediction[:,0:11], axis=1)[0]

    return predict_class_id


def predict_by_score_3(current_prediction):

    alfa = 3.0

    score_with_unknown    = np.max(current_prediction, axis=1)[0]
    score_without_unknown = np.max(current_prediction[:,0:11], axis=1)[0]
    score_unknown         = current_prediction[0,11]
    div_score             = score_without_unknown/(score_with_unknown + 0.00001)

    predict_class_id = -1

    if score_without_unknown < alfa:
        predict_class_id        = 11
    else:
        predict_class_id        = np.argmax(current_prediction[:,0:11], axis=1)[0]

    return predict_class_id



def  main():

    ############################################################
    # Import data
    testset   = load_data_test(config.DATADIR)
    test_gen  = data_generator_test(testset)

    print ('--------------------------------------------------')
    print ('Load predict set. Len = {} files'.format(len(testset)))
    print ('--------------------------------------------------')
    ############################################################

    
    conf_matrix = np.zeros(shape = [len(config.POSSIBLE_LABELS), len(config.POSSIBLE_LABELS)], dtype = float)

    ############################################################
    # Load NN

    lst_class_id   = []
    lst_prediction = []

    with tf.Session() as sess:
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('./model/' + config.RESTORE_MODEL_NAME)
        saver.restore(sess, tf.train.latest_checkpoint('./model'))

        graph = tf.get_default_graph()
        prediction = graph.get_tensor_by_name("model_deepnn/y:0")

        for i in range (len(testset)):
            print ('---------------------------')
            class_id, path, x_batch =  next(test_gen)
            current_prediction      = sess.run(prediction, feed_dict={"input/x:0": x_batch,
                                                                      "input/keep_prob:0": 1.0})

            predict_class_id        = predict_by_score_1(current_prediction)

            print (class_id, config.id2name[class_id], path, x_batch.shape)
            print (predict_class_id, config.id2name[predict_class_id], path, x_batch.shape)

            conf_matrix[class_id][predict_class_id] = conf_matrix[class_id][predict_class_id] + 1

            lst_class_id.append(class_id)
            lst_prediction.append(current_prediction)

    print ('---------------------------')
    np.set_printoptions(suppress=True)
    print (conf_matrix)
    print ('---------------------------')
 

    print ("Accuracy averall = {0}".format(sum(np.diag(conf_matrix)) / float (sum(sum(conf_matrix)))))
    for i in range(len(config.POSSIBLE_LABELS)):
        print ("    {:15s}  {:2f}".format(config.id2name[i], conf_matrix[i,i] / float (sum(conf_matrix[i,:]) + 0.001)))

    # Save prediction result for test
    lst_class_id   = np.array(lst_class_id)
    lst_prediction = np.array(lst_prediction)
    np.save('./out/_class_id',   lst_class_id)
    np.save('./out/_prediction', lst_prediction)




if __name__ == '__main__':
    main()
    
    
    


# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import pandas as pd

import config
from data_load import *
from data_generator import *


def  main():

    ############################################################
    # Import data
    predictset   = load_data_predict(config.DATADIR)
    predict_gen  = data_generator_predict(predictset)

    print ('--------------------------------------------------')
    print ('Load predict set. Len = {} files'.format(len(predictset)))
    print ('--------------------------------------------------')
    ############################################################

    ############################################################
    # Load NN

    with tf.Session() as sess:
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('./model/' + config.RESTORE_MODEL_NAME)
        saver.restore(sess, tf.train.latest_checkpoint('./model'))

        graph = tf.get_default_graph()
        prediction = graph.get_tensor_by_name("model_deepnn/y:0")

        l_fname = []
        l_label = []

        for i in range (len(predictset)):
            path, x_batch =  next(predict_gen)

            current_prediction      = sess.run(prediction, feed_dict={"input/x:0": x_batch,
                                                                      "input/keep_prob:0": 1.0})

            predict_class_id        = np.argmax(current_prediction, axis=1)[0]

            predict_label = config.id2name[predict_class_id]

            l_fname.append(os.path.basename(path))
            l_label.append(predict_label)

            if i%100 == 0:
                print ('    process {} files from {}'.format(i, len(predictset)))

        # Save to csv
        df = pd.DataFrame(columns=['fname', 'label'])
        df['fname'] = l_fname
        df['label'] = l_label
        df.to_csv(os.path.join(config.OUTDIR, r'sub.csv'), index=False)


if __name__ == '__main__':
    main()
    
    
    


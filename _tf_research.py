# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config
from data_load import *
from data_generator import *


def  main():

    np.set_printoptions(suppress=True)

    ############################################################
    # Load prediction result 
    lst_class_id    = np.load('./out/_class_id.npy')
    lst_predictions = np.load('./out/_prediction.npy')

    print ('--------------------------------------------------')
    print ('Load predict result. class_id.shape = {} predictions.shape = {}'.format(lst_class_id.shape, lst_predictions.shape))
    print ('--------------------------------------------------')
    ############################################################


    sc_target     = []
    sc_imposter   = []



    for i in range(len(lst_class_id)):
        class_id             = lst_class_id[i]
        current_prediction   = lst_predictions[i,0,:]

#        print ('---------------------------')
#        print (class_id)
#        print (current_prediction)

        score_with_unknown    = np.max(current_prediction, axis=0)
        score_without_unknown = np.max(current_prediction[0:11], axis=0)
        score_unknown         = current_prediction[11]

        div_score = score_without_unknown/(score_with_unknown + 0.00001)
        #div_score = score_without_unknown/(score_unknown + 0.00001)

        if config.id2name[class_id] == 'unknown':
            sc_imposter.append(div_score)
            #sc_imposter.append(score_with_unknown)
        else:
            sc_target.append(div_score)
            #sc_target.append(score_with_unknown)


    plt.hist(sc_target,   bins = 100, alpha=0.5, label='target')
    plt.hist(sc_imposter, bins = 100, alpha=0.5, label='imposter')

    plt.title("Score div histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
    
    
    


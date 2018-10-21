'''
Implemention of metrics: BER & BLER

Author: Lucyyang
'''
import tensorflow as tf
from keras import backend as K

def errors(y_true, y_pred):
    print(y_true,y_pred)
    return K.sum(tf.cast(K.not_equal(y_true, K.round(y_pred)),tf.int32))

def BER(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))
    return ber
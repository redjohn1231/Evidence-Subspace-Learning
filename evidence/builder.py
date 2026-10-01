import tensorflow as tf
from .FD import FD_basis
from .FSC import FSC_basis
from .GS import GS_basis

def build_basis(features,
                labels):

    real=tf.boolean_mask(features,
                         labels==4)

    fake=tf.boolean_mask(features,
                         labels!=4)
    
    Pa=FD_basis(real,
                fake)
    
    Pc=FSC_basis(features)

    Pg=GS_basis(features,
                labels)

    P=tf.concat([Pa,
                 Pc,
                 Pg],
                axis=1)

    with tf.device('/CPU:0'):
        P,_=tf.linalg.qr(P)

    return P

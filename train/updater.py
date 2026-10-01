# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:44:21 2026

@author: Thinker
"""

import tensorflow as tf


def EMA_update(evidence,
               P_new):
    
    Q_new = tf.matmul(P_new,
                      P_new,
                      transpose_b=True)

    Q = (evidence.beta * evidence.Q +
        (1.0 - evidence.beta) * Q_new)
    
    

    evidence.Q.assign(Q)


    with tf.device('/CPU:0'):

        eigval, eigvec = tf.linalg.eigh(Q)

    K = evidence.evidence_dim
    
    V = eigvec[:, -K:]

    A = tf.math.l2_normalize(P_new,
                             axis=0)


    M = tf.matmul(V,A,
        transpose_a=True)


    with tf.device('/CPU:0'):
        S, U , Vh = tf.linalg.svd(
            M,
            full_matrices=False)


    R = tf.matmul(U,Vh,
                  transpose_b=True)

    P = tf.matmul(V,R)

    similarity = tf.reduce_sum(P * A,axis=0)


    signs = tf.where(similarity >= 0.0,
                     tf.ones_like(similarity),
                     -tf.ones_like(similarity))

    P = P * signs[tf.newaxis, :]

    evidence.P.assign(P)
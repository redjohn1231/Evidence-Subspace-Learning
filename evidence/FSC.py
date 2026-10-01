import tensorflow as tf

def FSC_basis(features):


    X=tf.cast(features,
              tf.float32)

    X=X-tf.reduce_mean(X,
                       axis=0,
                       keepdims=True)

    X=X / (tf.math.reduce_std(X,axis=0,keepdims=True)+1e-6)

    A=tf.matmul(X,X,
                transpose_a=True)

    A=A/tf.cast(tf.shape(features)[0],
                tf.float32)

    A=tf.abs(A)

    A=A-tf.linalg.diag(tf.linalg.diag_part(A))

    D=tf.linalg.diag(tf.reduce_sum(A,axis=1))

    L=D-A

    with tf.device('/CPU:0'):
        eigval,eigvec=tf.linalg.eigh(L)

    p=eigvec[:,-1]

    return p[:,None]
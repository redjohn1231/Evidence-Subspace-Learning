import tensorflow as tf



def GS_basis(
        features,
        labels,
        num_basis=3):

    dim = features.shape[-1]

    fake_mask = labels < 4

    fake_features = tf.boolean_mask(features,
                                    fake_mask)
    
    fake_labels = tf.boolean_mask(labels,
                                  fake_mask)

    mu = tf.reduce_mean(fake_features,
                        axis=0)

    Sb = tf.zeros([dim,
                   dim],
        dtype=features.dtype)


    classes = tf.constant([0, 1, 2, 3],
                          dtype=fake_labels.dtype)

    for c in classes:

        idx = tf.where(fake_labels == c)

        fc = tf.gather_nd(fake_features,idx)

        muc = tf.reduce_mean(fc,axis=0)

        n = tf.cast(tf.shape(fc)[0],
                    features.dtype)

        d = muc - mu

        Sb += n * tf.tensordot(d,d,
                               axes=0)

    with tf.device('/CPU:0'):

        eigval, eigvec = tf.linalg.eigh(Sb)

    return eigvec[:, -num_basis:]
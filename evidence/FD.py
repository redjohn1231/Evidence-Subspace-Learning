import tensorflow as tf

def FD_basis(real,fake):

    mu_r=tf.reduce_mean(real,
                        axis=0)

    mu_f=tf.reduce_mean(fake,
                        axis=0)
    
    diff=mu_f-mu_r

    Xr=real-mu_r
    
    Xf=fake-mu_f

    Sw=tf.matmul(Xr,Xr,
                 transpose_a=True)

    Sw+=tf.matmul(Xf,Xf,
                  transpose_a=True)

    with tf.device('/CPU:0'):
        p=tf.matmul(
            tf.linalg.pinv(Sw),
            diff[:,None])
        
    return p


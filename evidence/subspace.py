import tensorflow as tf
class EvidenceSubspace:

    def __init__(self,
                 feature_dim,
                 evidence_dim,
                 beta=0.9):

        self.feature_dim=feature_dim

        self.evidence_dim=evidence_dim

        self.beta=beta

        P=tf.random.normal([feature_dim,
                            evidence_dim])

        P,_=tf.linalg.qr(P)


        self.P=tf.Variable(P,
                           trainable=False)

        self.Q=tf.Variable(
            tf.matmul(P,P,
                      transpose_b=True),
            trainable=False)

    # def projection(self,feature):

    #     return tf.matmul(feature, self.P)

import tensorflow as tf
from tensorflow.keras import layers

class EvidenceProjection(layers.Layer):
    def __init__(self,evidence):

        super().__init__()

        self.evidence=evidence

    def call(self,x):

        return tf.matmul(x,self.evidence.P)
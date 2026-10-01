from tensorflow import keras
from .evidence_layer import EvidenceProjection

class EBNet(keras.Model):

    def __init__(self,
                 backbone,
                 evidence,
                 classes):

        super().__init__()

        self.backbone=backbone
        self.evidence_layer=EvidenceProjection(evidence)
        self.fc=keras.layers.Dense(classes)
        
    def set_backbone_trainable(self,
                               trainable):

        self.backbone.trainable = trainable

        print("Backbone trainable:",
              self.backbone.trainable)

    def call(self,x):

        f = self.backbone(x)
        alpha=self.evidence_layer(f)
        
        return self.fc(alpha)

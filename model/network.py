import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
import random
SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.keras.utils.set_random_seed(SEED)


def build_resnet50_backbone(
    input_shape=(256, 256, 3),
    output_layer_name="conv4_block6_out",  
    trainable=True,
    w_name=None):

    inputs = tf.keras.Input(shape=input_shape)


    base_model = ResNet50(
        include_top=False,
        weights=w_name,
        input_tensor=inputs)
    
    base_model.trainable = True

    try:
        feature_map = base_model.get_layer(output_layer_name).output
    except ValueError:
        raise ValueError(f"Layer {output_layer_name} not found in ResNet50.")
    
    features = GlobalAveragePooling2D(name="gap")(feature_map)
    
    model = Model(inputs=inputs, outputs=features,name="feature_extractor")

    return model


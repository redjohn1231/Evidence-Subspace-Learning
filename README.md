# Evidence Subspace Learning for Interpretable Deepfake Detection

In main.py, the dataset module provides the standard data generators used for training, validation, and testing. The dataset module is responsible for loading and preprocessing the dataset and generating the corresponding training, validation, and test sets. The data generators should be configured according to the actual dataset organization and local data paths before running the code.

The network weights and evidence Bases are saved separately. The network weights are stored in `model.weights.h5`, while the evidence Bases is stored in `P.npy`. During inference, both files must be loaded to reconstruct the complete detection model.

1. train/ — Training and Parameter Optimization

This directory contains the training scripts and configurations responsible for optimizing the network parameters and updating the evidence bases.

2. model/ — Network Architecture

This directory contains the complete implementation of the proposed Evidence Basis-based Deepfake Detection Framework.

3. evidence/ — Forensic Attribute Construction

This directory contains the implementations for constructing the three forensic attributes.

## Requirements

- **Python**: 3.9 
- **TensorFlow**: 2.9.1  
- **Keras**: 2.9.0 (included in TensorFlow 2.9.1)  

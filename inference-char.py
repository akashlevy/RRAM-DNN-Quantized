# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Setup
# 
# ## Import TensorFlow and NumPy

# %%
# Import libraries
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import argparse

parser = argparse.ArgumentParser(description='Inference characterization.')
parser.add_argument('i', type=int, help='i value to process')
args = parser.parse_args()

# %% [markdown]
# ## Configure DNN settings
# 
# Here, we specify the ResNet architecture parameters:

# %%
# Number of classes to infer
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Depth parameter
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# %% [markdown]
# ## Load dataset and preprocess
# 
# We are working with the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) here.

# %%
# Load the CIFAR10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions
input_shape = x_train.shape[1:]

# Normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# %% [markdown]
# ## Load trained ResNet model

# %%
# Load model
model_name = 'weight_quant_4b_final_3'

# Model name, depth, and version
model_type = 'ResNet%dv%d_%s' % (depth, version, model_name)
print(model_type)

# Prepare model saving directory
save_dir = os.path.join(os.getcwd(), model_type)
model_full_name = 'cifar10_%s_model' % model_type
filepath = os.path.join(save_dir, model_full_name)

# Load model checkpoint
K.clear_session()
model = load_model(filepath)


# %%
# Save all model parameters to a dict
weights = {}
for olayer in model.layers:
    weights[olayer.name] = olayer.get_weights()


# %%
# Perform baseline inference before quantization
model.evaluate(x_test, y_test, verbose=1)


# %%
# Perform quantized inference
weights_quant = {}
weights_enci = {}
for layer_name in weights:
    if 'conv2d' in layer_name or 'dense' in layer_name:
        # Get weights from layer
        print(layer_name)
        W, b, W_max = weights[layer_name]

        # Get quantized weights
        weights_quant[layer_name] = tf.quantization.fake_quant_with_min_max_args(W, -W_max, W_max, 4, narrow_range=True).numpy()
        
        # Get quantized weights encoding index
        weights_enci[layer_name] = np.int8(np.round(weights_quant[layer_name] / W_max * 7))
        positivize = lambda x : x + 1 if x > 0 else x
        weights_enci[layer_name] = np.vectorize(positivize)(weights_enci[layer_name]) + 7
        
        print(sorted(np.unique(weights_enci[layer_name])))


# %%
# Create weight matrices after RRAM relaxation according to the confusion matrices (4 levels per cell)
for seed in range(1):
    # Set random seed for statistics
    np.random.seed(seed)

    # Sweep indices
    i = args.i
    for j in [14,15]:
        # Ignore diagonal entries
        if i == j:
            continue

        # Sweep BERs
        for ber in 10.**np.linspace(-3, 0, 17):
            # Log
            print(f'Doing {seed}\t{i}\t{j}\t{ber}')

            # Create confusion matrix
            C = np.eye(16)
            C[i][i] = 1-ber
            C[i][j] = ber

            # Perturb weights under confmat error model
            weights_perturb = {}
            for layer_name in weights:
                if 'conv2d' in layer_name or 'dense' in layer_name:
                    W, b, W_max = weights[layer_name]

                    # Perturb based on confusion matrix
                    perturb = lambda x : np.random.choice(16, p=C[x])
                    weights_perturb[layer_name] = np.vectorize(perturb)(weights_enci[layer_name])

                    # Scale back to weight value
                    depositivize = lambda x : x - 1 if x > 7 else x
                    weights_perturb[layer_name] = (np.vectorize(depositivize)(weights_perturb[layer_name]) - 7) * W_max / 7

            # Load relaxed weights back to the model
            for layer in model.layers:
                if layer.name in weights_perturb:
                    W, b, W_max = layer.get_weights()
                    layer.set_weights([weights_perturb[layer.name], b, W_max])

            # Evaluate accuracy after relaxation
            loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

            # Append results to file
            with open(f'out/char-{i}.tsv', 'a') as outf:
                print(f'{seed}\t{i}\t{j}\t{ber}\t{loss}\t{accuracy}\n')
                outf.write(f'{seed}\t{i}\t{j}\t{ber}\t{loss}\t{accuracy}\n')

# %% [markdown]
# TODO: get weight freqs


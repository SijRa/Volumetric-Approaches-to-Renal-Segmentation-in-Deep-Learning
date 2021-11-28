from utils.data_loader import DataLoader
from utils.data_generator import DataGenerator
from utils.callbacks import Plateau_Decay, Early_Stopping
from utils.model import _3dUNet, _3dUNetPlusPlusL2, _3dUNetPlusPlusL3

from sklearn.model_selection import KFold

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid")

import sys


# Models
models = [_3dUNetPlusPlusL2, _3dUNetPlusPlusL3, _3dUNet]

# Model params
epochs = 40
learning_rate = 0.0005
batch_size = 1

k_folds = 5

# Data loading params
data_limit = None
height = 240
width = 240
depth = 11
n_channels = 1
params = {"batch_size": batch_size, 
        "dim": (height, width, depth), 
        "n_channels": n_channels}


# dictionary for all results
results = {}
time_taken = {}

# Functionality
saveResults = False

def save_results():
    with open("results.txt", "w") as f:
        for key, nested in sorted(results.items()):
            print(key, file=f)
            print('   Train', file=f)
            for subkey, value in sorted(nested.items()):
                print('   {}: {}'.format(subkey, value.history["dice_coef"]), file=f)
            print('   Test', file=f)
            for subkey, value in sorted(nested.items()):
                print('   {}: {}'.format(subkey, value.history["val_dice_coef"]), file=f)
            print(file=f)
        for key, nested in sorted(time_taken.items()):
            print(key, file=f)
            for subkey, value in sorted(nested.items()):
                print('   {}: {}'.format(subkey, value), file=f)
            print(file=f)

def main(data_limit=None):
    # Directories for data folders
    data_dir = "dataset/"
    labels_filename = "Subject_Information.xlsx"
    data_limit = data_limit
    data_loader = DataLoader(data_dir, labels_filename, data_limit)
    partition, labels = data_loader.load_data()
    
    # Generators
    #training_generator = DataGenerator(partition['train'], data_dir, labels, **params)
    #validation_generator = DataGenerator(partition['validation'], data_dir, labels, **params)
    
    # Callbacks
    monitor = 'val_loss'
    callbacks = [Plateau_Decay(monitor, factor=0.8, patience=3)]
  
    fold_generator = KFold(n_splits=k_folds, shuffle=True)
    
    with tf.device("/GPU:1"):
        for Model in models:
            fold = 0
            # initalise model
            model = Model(input_shape=(height, width, depth, n_channels), learning_rate=learning_rate)
            print(model.summary())
            initial_weight_values = model.get_weights() # save weights
            # initialise dictionaries for folds
            results[Model.__name__] = {}
            time_taken[Model.__name__] = {}
            for i in range(k_folds):
                results[Model.__name__][i+1] = None
                time_taken[Model.__name__][i+1] = None
            # K-Fold training
            for train_index, validation_index in fold_generator.split(partition['all']):
                fold += 1
                model.set_weights(initial_weight_values) # reinitialise weights
                train = np.take(partition['all'], train_index, axis=0)
                validation = np.take(partition['all'], validation_index, axis=0)
                training_generator = DataGenerator(train, data_dir, labels, **params)
                validation_generator = DataGenerator(validation, data_dir, labels, **params)
                start = time.time() # measure time
                training_results = model.fit(training_generator,
                  epochs=epochs,
                  validation_data=validation_generator,
                  callbacks=callbacks)
                end = time.time()
                train_time = (end-start)/60
                time_taken[Model.__name__][fold] = train_time
                results[Model.__name__][fold] = training_results
        if saveResults:
            save_results()

# Check for args
if __name__ == "__main__":
    if len(sys.argv) > 1:
    	main(int(sys.argv[1]))
    else:
        main()
    sys.exit(100)
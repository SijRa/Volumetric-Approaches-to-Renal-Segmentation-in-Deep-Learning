import sys
import os

import numpy as np
import nibabel as nib

from tensorflow import keras
from scipy.stats import zscore
from utils.preprocessing import fuzzy_normalisation


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_dir, labels, batch_size=1, dim=(240,240,11), n_channels=1, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __extract_mri__(self, ID):
        try:
            return nib.load(self.data_dir + ID + "/" + 'T2_corrected.nii.gz')
        except FileNotFoundError:
            return nib.load(self.data_dir + ID + "/" + 'T2_V1.nii.gz')
    
    def __extract_mask__(self, ID):
        try:
            return nib.load(self.data_dir + ID + "/" + 'T2_mask.nii.gz')
        except FileNotFoundError:
            return nib.load(self.data_dir + ID + "/" + 'T2_V1_mask.nii.gz')

    def __extract_mri_R__(self, ID, RTag):
        return nib.load(self.data_dir + ID + "/" + 'T2_' + RTag + '_corrected.nii.gz')
    
    def __extract_mask_R__(self, ID, RTag):
        return nib.load(self.data_dir + ID + "/" + 'T2_' + RTag + '_mask.nii.gz')
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))
        #print(list_IDs_temp)    
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            RTag = ID.split("_")[-1]
            mri = None
            mask = None
            if "R" in RTag:
                ID = "_".join(ID.split("_")[:-1])
                mri = self.__extract_mri_R__(ID, RTag)
                mask = self.__extract_mask_R__(ID, RTag)
            else:
                mri = self.__extract_mri__(ID)
                mask = self.__extract_mask__(ID)
            # Preprocessing
            #mri = fuzzy_normalisation(mri, mask).get_fdata()
            mri = zscore(mri.get_fdata(), axis=None)
            X[i,] = mri[:,:,:11].reshape((*self.dim, self.n_channels))
            y[i,] = mask.get_fdata()[:,:,:11]
        return X, y
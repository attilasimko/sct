from __future__ import print_function
from comet_ml import Experiment
import os
import numpy as np
from tensorflow import experimental
from tensorflow.keras import optimizers, metrics, layers
from tensorflow.keras.backend import clear_session
import random
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from scipy.ndimage import shift
import numpy as np
import argparse
import gc
import matplotlib.pyplot as plt
import tensorflow
from numpy.random import seed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import sys
sys.path.append("../")
import MLTK
from MLTK.models import sCTNet2
from MLTK.data import DataGenerator
pid = os.getpid()
random.seed(2021)
seed(2021)
print(pid)

data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0052/'
exps = ["t1w", "t2w", "ft1w", "ft2w"]
gen_train = DataGenerator(data_path + 'fulln',
                    inputs=[['id', False, 'int'],
                            [exps[0], True, 'float32'],
                            [exps[1], True, 'float32'],
                            [exps[2], True, 'float32'],
                            [exps[3], True, 'float32']],
                    outputs=[['mask', False, 'uint8'],
                            ['slice', False, 'int']],
                    batch_size=1,
                    shuffle=True)




for pat_idx in range(1, 20):
    t1w = np.zeros((120, 256, 256))
    t2w = np.zeros((120, 256, 256))
    st1w = np.zeros((120, 256, 256))
    st2w = np.zeros((120, 256, 256))
    slice_max = 0
    for idx in range(len(gen_train)):
        x, y = gen_train[idx]
        if (x[0][0, 0] == pat_idx):
            t1w[y[1][0, 0], :, :] = x[1][0, :, :, 0]
            t2w[y[1][0, 0], :, :] = x[2][0, :, :, 0]
            st1w[y[1][0, 0], :, :] = x[3][0, :, :, 0]
            st2w[y[1][0, 0], :, :] = x[4][0, :, :, 0]
            slice_max = np.max([slice_max, y[1][0, 0]])
    t1w = t1w[0:slice_max, :, :]
    t2w = t2w[0:slice_max, :, :]
    st1w = st1w[0:slice_max, :, :]
    st2w = st2w[0:slice_max, :, :]

    plt.subplot(2, 2, 1)
    plt.imshow(t1w[:, :, 150], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(t2w[:, :, 150], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.imshow(st1w[:, :, 150], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 4)
    plt.imshow(st2w[:, :, 150], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
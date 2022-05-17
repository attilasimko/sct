from __future__ import print_function
import os
from sympy import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Code suppressing TF warnings and messages.
import numpy as np
from tensorflow import experimental
from tensorflow.keras import optimizers, metrics, layers
from tensorflow.keras.backend import clear_session
import random
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
from tensorflow import function, TensorSpec
from tensorflow import io
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import sys
from models import build_discriminator, build_srresnet, build_unet
from utils import get_patients, custom_loss
from data import DataGenerator
pid = os.getpid()
random.seed(2021)
seed(2021)
print(pid)

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--lr_dis", default=0.00005, help="Learning rate for generator.")
parser.add_argument("--lr_gan", default=0.00005, help="Learning rate for generator.")
parser.add_argument("--alpha", default=0.0001, help="Learning rate for generator.")
parser.add_argument("--batch_size", default=4, help="Learning rate for generator.")
parser.add_argument("--case", default=7, help="Learning rate for generator.")
parser.add_argument("--gpu", default=None, help="Learning rate for generator.")
parser.add_argument("--inarray", default="", help="Learning rate for generator.")
parser.add_argument("--outarray", default="", help="Learning rate for generator.")

args = parser.parse_args()
comet = 'sct'

batchnorm = "True"
num_filters = 64
num_res_block = 12

gpu = args.gpu
lr_dis = float(args.lr_dis)
lr_gan = float(args.lr_gan)
alpha = float(args.alpha)
case = int(args.case)
batch_size = int(args.batch_size)

data_path = "" # Path to dataset




gen_train = DataGenerator(data_path + 'training_' + str(case),
                    inputs=[['mr', False, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=True)

gen_val = DataGenerator(data_path + 'validating',
                    inputs=[['mr', False, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=1,
                    shuffle=False)

gen_test_baseline = DataGenerator(data_path + 'testing_baseline',
                    inputs=[['mr', False, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=1,
                    shuffle=False)

gen_test_abnormal = DataGenerator(data_path + 'testing_abnormal',
                    inputs=[['mr', False, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=1,
                    shuffle=False)

gen_test_female = DataGenerator(data_path + 'testing_female',
                    inputs=[['mr', False, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=1,
                    shuffle=False)

gen_test_t1w = DataGenerator(data_path + 'testing_t1w',
                    inputs=[['mr', False, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=1,
                    shuffle=False)

generator = build_srresnet(num_filters=num_filters, batchnorm=batchnorm)
generator.compile(optimizer=optimizers.Adam(0.001), loss=["mse"], run_eagerly=True)
discriminator = build_discriminator(64, 512, 512, 1)  
discriminator.compile(loss=["mse"], metrics=["accuracy"], optimizer=optimizers.Adam(lr_dis), run_eagerly=True)


# GAN
img_in = Input(shape=(512, 512, 1), batch_size=batch_size)
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
pred_mt = generator(img_in)
pred_d = discriminator(pred_mt)
GAN = Model(inputs=img_in, outputs=[pred_d, pred_mt])
losses = {
    "model_1": "mse", # GAN
    "model": "mse" # PAIRED
}
GAN.compile(optimizer=optimizers.Adam(lr_gan), loss=losses, metrics=['accuracy'], loss_weights=[alpha, 1], run_eagerly=True) # binary_crossentropy
patience_thr = 20
overall = []

real_label = np.ones((batch_size, 1))
fake_label = np.zeros((batch_size, 1))

patience = 0
best_loss = np.inf
n_epochs = 200
go_gan = False
ct_thr = 0

validation_patients = get_patients(gen_val)
training_patients = get_patients(gen_train)

models = [] # List of paths to the pre-trained .h5 files.

def test_gen(gen):
    a_list = []
    st_list = []
    b_list = []
    for idx in range(len(gen)):
        x_mri, x_ct = gen[idx]

        pred = generator.predict_on_batch(x_mri[0])
        sct = pred * 1000
        ct = x_ct[0] * 1000
        a_list.append(np.average(np.abs(sct - ct), weights=((ct > -1000) * (ct <= -100))))
        st_list.append(np.average(np.abs(sct - ct), weights=((ct > -100) * (ct <= 100))))
        b_list.append(np.average(np.abs(sct - ct), weights=((ct > 100) * (ct <= 1000))))

    res1 = str(np.round(np.nanmean(a_list), 5)) + " +- " + str(np.round(np.nanstd(a_list) / np.sqrt(len(a_list)), 5))
    res2 = str(np.round(np.nanmean(st_list), 5)) + " +- " + str(np.round(np.nanstd(st_list) / np.sqrt(len(st_list)), 5))
    res3 = str(np.round(np.nanmean(b_list), 5)) + " +- " + str(np.round(np.nanstd(b_list) / np.sqrt(len(b_list)), 5))
    print(res1 + " " + res2 + " " + res3)


for model in models:
    print("\n"+model)

    generator.load_weights(model)
    print("Baseline:")
    test_gen(gen_test_baseline)

    print("Abnormal:")
    test_gen(gen_test_abnormal)

    print("Female:")
    test_gen(gen_test_female)

    print("T1W:")
    test_gen(gen_test_t1w)
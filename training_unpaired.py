from __future__ import print_function
import os
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
parser.add_argument("--case", default=1, help="Learning rate for generator.")
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
generator.compile(optimizer=optimizers.Adam(lr_gan), loss=["mse"], run_eagerly=True)

patience_thr = 50
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

for epoch in range(n_epochs):
    tensorflow.keras.backend.clear_session()

    MAE_training = []
    MAE_testing = []

    d_loss_list = []
    gan_loss_list = []
    d_mri_loss_list = []
    d_ct_loss_list = []
    ct_loss_list = []
    ct_masked_loss_list = []
    for idx in range(len(gen_train)):
        x_mri, x_ct = gen_train[idx]

        pred = generator.predict_on_batch(x_mri[0])

        gan_loss = generator.train_on_batch(x_mri[0], x_ct[0])
        gan_loss_list.append(gan_loss)
    gen_train.on_epoch_end()

    for idx in range(len(gen_val)):
        x_mri, x_ct = gen_val[idx]

        pred = generator.predict_on_batch(x_mri[0])

        ct_loss_list.append(np.average(np.abs(pred - x_ct[0])))
        ct_masked_loss_list.append(np.average(np.abs(pred - x_ct[0]), weights=x_ct[0] > -1))
    gen_val.on_epoch_end()


    if (best_loss > (np.mean(ct_masked_loss_list))):
        patience = 0
        best_loss = (np.mean(ct_masked_loss_list))
        scores = []
        MAE_testing = []
        count = 0
        inp = []
        target = []
        mask = []
        kloss = []
        for patient in validation_patients:
            data = np.zeros((500, 512, 512, 1))
            ct_data = np.zeros((500, 512, 512, 1))
            pred = np.zeros((500, 512, 512, 1))
            plot_z = 0
            gen_val.on_epoch_end()
            for idx in range(len(gen_val)):
                if patient == gen_val.file_list[idx].split('/')[-1].split('_')[0]:
                    x_mri, x_ct = gen_val[idx]
                    slc = int(gen_val.file_list[idx].split('_')[-2])
                    if (int(gen_val.file_list[idx].split('_')[-1].split('.')[0]) == 0):
                        data[slc, :, :, 0] = x_mri[0][0, :, :, 0]
                        ct_data[slc, :, :, 0] = x_ct[0][0, :, :, 0]
                        pred[slc, :, :, 0] = generator.predict_on_batch(x_mri[0])[0, :, :, 0]
                        plot_z += 1


            data = data[0:plot_z, :, :, :]
            ct_data = ct_data[0:plot_z, :, :, :]
            pred = pred[0:plot_z, :, :, :]

            plt.figure(figsize=(20, 12))
            plt.subplot(3, 3, 1)
            plt.imshow(data[:, 250, :, 0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 3, 2)
            plt.imshow(pred[:, 250, :, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 3, 3)
            plt.imshow(ct_data[:, 250, :, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 3, 4)
            plt.imshow(data[:, :, 150, 0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 3, 5)
            plt.imshow(pred[:, :, 150, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 3, 6)
            plt.imshow(ct_data[:, :, 150, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 3, 7)
            plt.imshow(data[15, :, :, 0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 3, 8)
            plt.imshow(pred[15, :, :, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 3, 9)
            plt.imshow(ct_data[15, :, :, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.close('all')

        generator.save_weights(str(out_path) + str(pid) + '.h5')
        full_model = function(lambda x: generator(x)) 
        full_model = full_model.get_concrete_function(TensorSpec(generator.inputs[0].shape, generator.inputs[0].dtype))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=out_path,
                        name=f'{pid}_frozen.pb',
                        as_text=False)
    else:
        patience += 1
        if (patience == patience_thr):
            break


generator.load_weights(str(out_path) + str(pid) + '.h5')

a_list = []
st_list = []
b_list = []
for idx in range(len(gen_test_baseline)):
    x_mri, x_ct = gen_test_baseline[idx]

    pred = generator.predict_on_batch(x_mri[0])

    a_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > -1) * (x_ct[0] <= -0.1))))
    st_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > -0.1) * (x_ct[0] <= 0.1))))
    b_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > 0.1) * (x_ct[0] <= 1))))

print("\nBaseline:")
print(str(np.round(np.nanmean(a_list), 5)) + " +- " + str(np.round(np.nanstd(a_list) / np.sqrt(len(a_list)), 10)))
print(str(np.round(np.nanmean(st_list), 5)) + " +- " + str(np.round(np.nanstd(st_list) / np.sqrt(len(st_list)), 10)))
print(str(np.round(np.nanmean(b_list), 5)) + " +- " + str(np.round(np.nanstd(b_list) / np.sqrt(len(b_list)), 10)))

a_list = []
st_list = []
b_list = []
for idx in range(len(gen_test_abnormal)):
    x_mri, x_ct = gen_test_abnormal[idx]

    pred = generator.predict_on_batch(x_mri[0])

    a_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > -1) * (x_ct[0] <= -0.1))))
    st_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > -0.1) * (x_ct[0] <= 0.1))))
    b_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > 0.1) * (x_ct[0] <= 1))))


print("\nAbnormal:")
print(str(np.round(np.nanmean(a_list), 5)) + " +- " + str(np.round(np.nanstd(a_list) / np.sqrt(len(a_list)), 10)))
print(str(np.round(np.nanmean(st_list), 5)) + " +- " + str(np.round(np.nanstd(st_list) / np.sqrt(len(st_list)), 10)))
print(str(np.round(np.nanmean(b_list), 5)) + " +- " + str(np.round(np.nanstd(b_list) / np.sqrt(len(b_list)), 10)))

a_list = []
st_list = []
b_list = []
for idx in range(len(gen_test_female)):
    x_mri, x_ct = gen_test_female[idx]

    pred = generator.predict_on_batch(x_mri[0])

    a_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > -1) * (x_ct[0] <= -0.1))))
    st_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > -0.1) * (x_ct[0] <= 0.1))))
    b_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > 0.1) * (x_ct[0] <= 1))))


print("\nFemale:")
print(str(np.round(np.nanmean(a_list), 5)) + " +- " + str(np.round(np.nanstd(a_list) / np.sqrt(len(a_list)), 10)))
print(str(np.round(np.nanmean(st_list), 5)) + " +- " + str(np.round(np.nanstd(st_list) / np.sqrt(len(st_list)), 10)))
print(str(np.round(np.nanmean(b_list), 5)) + " +- " + str(np.round(np.nanstd(b_list) / np.sqrt(len(b_list)), 10)))

a_list = []
st_list = []
b_list = []
for idx in range(len(gen_test_t1w)):
    x_mri, x_ct = gen_test_t1w[idx]

    pred = generator.predict_on_batch(x_mri[0])

    a_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > -1) * (x_ct[0] <= -0.1))))
    st_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > -0.1) * (x_ct[0] <= 0.1))))
    b_list.append(np.average(np.abs(pred - x_ct[0]), weights=((x_ct[0] > 0.1) * (x_ct[0] <= 1))))


print("\nT1W:")
print(str(np.round(np.nanmean(a_list), 5)) + " +- " + str(np.round(np.nanstd(a_list) / np.sqrt(len(a_list)), 10)))
print(str(np.round(np.nanmean(st_list), 5)) + " +- " + str(np.round(np.nanstd(st_list) / np.sqrt(len(st_list)), 10)))
print(str(np.round(np.nanmean(b_list), 5)) + " +- " + str(np.round(np.nanstd(b_list) / np.sqrt(len(b_list)), 10)))
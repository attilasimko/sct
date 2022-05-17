from __future__ import print_function
from comet_ml import Experiment
import os
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
import sys
sys.path.append("../")
import MLTK
from MLTK.synthetic_ct.models import sCTNet, build_discriminator, build_srresnet
from MLTK.data import DataGenerator
pid = os.getpid()
random.seed(2021)
seed(2021)
print(pid)

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--lr_dis", default=0.00001, help="Learning rate for generator.")
parser.add_argument("--lr_gan", default=0.00001, help="Learning rate for generator.")
parser.add_argument("--batch_size", default=8, help="Learning rate for generator.")
parser.add_argument("--update_idx", default=1.0, help="Learning rate for generator.")
parser.add_argument("--gpu", default=None, help="Learning rate for generator.")
parser.add_argument("--inarray", default="", help="Learning rate for generator.")
parser.add_argument("--outarray", default="", help="Learning rate for generator.")

args = parser.parse_args()
comet = 'sct'

gpu = args.gpu
lr_dis = float(args.lr_dis)
lr_gan = float(args.lr_gan)
update_idx = float(args.update_idx)
batch_size = int(args.batch_size)

data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0058/'
out_path = '/home/attilasimko/Documents/out/'
if gpu is not None:
    base = 'gauss'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    data_path = '/home/attila/data/DS0058/'
    out_path = '/home/attila/out/'
else:
    physical_devices = tensorflow.config.list_physical_devices('GPU')
    tensorflow.config.set_visible_devices(physical_devices[0], 'GPU')

# dev = tensorflow.config.list_physical_devices("GPU")
# print(dev)
# tensorflow.config.experimental.set_memory_growth(dev[1], True)

def mask_loss(y_true, y_pred, mask):
    diff = np.abs(y_true - y_pred)
    return np.average(diff, weights=mask)

if comet is not None:
    experiment = Experiment(api_key="ro9UfCMFS2O73enclmXbXfJJj",
                            project_name=str(comet),
                            workspace="attilasimko")
    experiment.log_parameters({"gpu":gpu,
                            "lr_dis":lr_dis,
                            "lr_gan":lr_gan,
                            "pid":pid,
                            "batch_size":batch_size})
else:
    experiment = False

gen_train_p = DataGenerator(data_path + 'training_p',
                    inputs=[['mr', False, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=True)


gen_train_mr = DataGenerator(data_path + 'training/MR',
                    inputs=[['data', False, 'float32']],
                    outputs=[],
                    batch_size=batch_size,
                    shuffle=True)

gen_train_ct = DataGenerator(data_path + 'training/CT',
                    inputs=[['data', False, 'float32']],
                    outputs=[],
                    batch_size=batch_size,
                    shuffle=True)

gen_val = DataGenerator(data_path + 'validating',
                    inputs=[['mr', False, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=1,
                    shuffle=False)

# gen_test = DataGenerator(data_path + 'testing',
#                     inputs=[['id', False, 'int'],
#                             ['t1w', False, 'float32'],
#                             ['slice', False, 'int']],
#                     outputs=[['ct', False, 'float32'],
#                             ['mask', False, 'uint8']],
#                     batch_size=1,
#                     shuffle=False)

generator = build_srresnet(num_filters=32, num_res_blocks=9) # sCTNet(8, 3, 0.2) # build_srresnet(num_filters=32, num_res_blocks=12)#sCTNet(8, 3, 0) # init_w is for sCTNet(16, 3, 0)
generator.compile(loss=["mae"], optimizer=optimizers.Adam(0.0005))
generator.load_weights(out_path + 'init.h5')
discriminator = build_discriminator(64, 512, 512, 1)
discriminator.compile(loss=["mse"], metrics=["accuracy"], optimizer=optimizers.Adam(lr_dis))

print('Generator with # params: ' + str(generator.count_params()))
print('Discriminator with # params: ' + str(discriminator.count_params()))

# GAN
img_in = Input(shape=(512, 512, 1), batch_size=batch_size)
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
pred_mt = generator([img_in])
pred_unit = generator(pred_mt)
pred_d = discriminator([pred_mt])
GAN = Model(inputs=img_in, outputs=pred_d)
GAN.compile(optimizer=optimizers.Adam(lr_gan), loss=['mse'], metrics=['accuracy']) # binary_crossentropy
# generator.save_weights('init_w')
experiment.log_parameters({ "gen_param": generator.count_params(),
                            "dis_param": discriminator.count_params(),
                            "lr_dis": lr_dis,
                            "lr_gan": lr_gan})
patience_thr = 50
overall = []

real_label = np.ones((batch_size, 1))
fake_label = np.zeros((batch_size, 1))

patience = 0
best_loss = np.inf
n_epochs = 500
go_gan = False
ct_thr = 0

validation_patients = []
for idx in range(len(gen_val)):
    patient = "_".join(gen_val.file_list[idx].split('/')[-1].split('_')[0:2])
    if validation_patients.count(patient) == 0:
        validation_patients.append(patient)

for epoch in range(n_epochs):
    experiment.set_epoch(epoch)

    MAE_training = []
    MAE_testing = []


    # K-fold training
    d_loss_list = []
    gan_loss_list = []
    d_mri_loss_list = []
    d_ct_loss_list = []
    ct_loss_list = []
    ct_masked_loss_list = []
    for idx in range(len(gen_train_p)):
        x_mri, x_ct = gen_train_p[idx]
        generator.train_on_batch(x_mri[0], x_ct[0]) 

    gen_train_mr.on_epoch_end()
    gen_train_ct.on_epoch_end()
                            
    for idx in range(len(gen_val)):
        x_mri, x_ct = gen_val[idx]
        
        pred = generator.predict_on_batch(x_mri[0])
        d_mri_loss = discriminator.test_on_batch(pred, fake_label[0:1])
        d_mri_loss_list.append(d_mri_loss[1])

        d_ct_loss = discriminator.test_on_batch(x_ct[0], real_label[0:1])
        d_ct_loss_list.append(d_ct_loss[1])

        ct_loss_list.append(np.average(np.abs(pred - x_ct[0])))
        ct_masked_loss_list.append(np.average(np.abs(pred - x_ct[0]), weights=x_ct[0] > -1))

    if ((not(go_gan)) & (np.mean(d_ct_loss_list) > 0.8) & (np.mean(d_mri_loss_list) > 0.8)):
        go_gan = True
        print("Go GAN.")

    # gen_val.on_epoch_end()
    
    experiment.log_metrics({"d_loss":np.round(np.mean(d_loss_list), 10),
                            "gan_loss":np.round(np.mean(gan_loss_list), 10),
                            "acc_fake":np.round(np.mean(d_mri_loss_list), 10),
                            "acc_real":np.round(np.mean(d_ct_loss_list), 10),
                            "ct_loss":np.round(np.mean(ct_loss_list), 10),
                            "ct_masked_loss":np.round(np.mean(ct_masked_loss_list), 10)})
                            
    if (10 > np.mean(ct_masked_loss_list)):
        best_loss = np.mean(ct_masked_loss_list)
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
            plot_z = 0
            gen_val.on_epoch_end()
            for idx in range(len(gen_val)):
                if patient in gen_val.file_list[idx]:
                    x_mri, x_ct = gen_val[idx]
                    # ct_data[x[2][0, 0], :, :, 0] = y[0][0, :, :, 0]
                    data[int(gen_val.file_list[idx].split('_')[-1].split('.')[0]), :, :, 0] = x_mri[0][0, :, :, 0]
                    ct_data[int(gen_val.file_list[idx].split('_')[-1].split('.')[0]), :, :, 0] = x_ct[0][0, :, :, 0]
                    plot_z += 1


            data = data[0:plot_z, :, :, :]
            ct_data = ct_data[0:plot_z, :, :, :]
            if (plot_z > 64):
                data = data[0:64, :, :, :]
                ct_data = ct_data[0:64, :, :, :]
            # ct_data = ct_data[0:plot_z, :, :, :]
            pred = generator.predict_on_batch(data)
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
            experiment.log_figure(figure=plt, figure_name=patient, overwrite=False, step=epoch)
            plt.close('all')
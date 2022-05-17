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
from numpy.random import seed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import sys
sys.path.append("../")
import MLTK
from MLTK.models import sCTNet
from MLTK.data import DataGenerator
pid = os.getpid()
random.seed(2021)
seed(2021)
print(pid)

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--lr", default=0.0005, help="Learning rate for generator.")
parser.add_argument("--batch_size", default=32, help="Learning rate for generator.")
parser.add_argument("--gpu", default=None, help="Learning rate for generator.")
parser.add_argument("--inarray", default="t1w", help="Learning rate for generator.")
parser.add_argument("--outarray", default="", help="Learning rate for generator.")

args = parser.parse_args()
comet = 'sct'

gpu = args.gpu
inarray = args.inarray
outarray = args.outarray
lr = float(args.lr)
batch_size = int(args.batch_size)

data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0052/'
out_path = '/home/attilasimko/Documents/out/'
if gpu is not None:
    base = 'gauss'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    data_path = '/home/attila/data/DS0052/'
    out_path = '/home/attila/out/' + str(gpu) + '/'

if comet is not None:
    experiment = Experiment(api_key="ro9UfCMFS2O73enclmXbXfJJj",
                            project_name=str(comet),
                            workspace="attilasimko")
    experiment.log_parameters({"gpu":gpu,
                            "learning rate":lr,
                            "pid":pid,
                            "inarray":inarray,
                            "outarray":outarray,
                            "batch_size":batch_size})
else:
    experiment = False

gen_train = DataGenerator(data_path + 'full',
                    inputs=[['id', False, 'int'],
                            [inarray, True, 'float32']],
                    outputs=[['ct', False, 'float32']],
                    batch_size=1,
                    shuffle=True)

gen_val = DataGenerator(data_path + 'full',
                    inputs=[['id', False, 'int'],
                            [inarray, True, 'float32'],
                            ['slice', False, 'int']],
                    outputs=[['ct', False, 'float32'],
                            ['mask', False, 'uint8']],
                    batch_size=1,
                    shuffle=True)

gen_test = DataGenerator(data_path + 'full',
                    inputs=[['id', False, 'int'],
                            ['t1w', True, 'float32'],
                            ['t2w', True, 'float32'],
                            ['ft1w', True, 'float32'],
                            ['slice', False, 'int']],
                    outputs=[['ct', False, 'float32'],
                            ['mask', False, 'uint8']],
                    batch_size=1,
                    shuffle=True)
exps = ["t1w", "t2w", "ft1w"]

model = sCTNet(16, 3, 0) # init_w is for sCTNet(16, 3, 0)
# model.save_weights('init_w')
experiment.log_parameters({ "parameters": model.count_params()})
patience_thr = 100
overall = []

def mask_loss(y_true, y_pred, mask):
    diff = np.abs(y_true - y_pred)
    return np.average(diff, weights=mask)

# folds = [[1, 10, 18], [2, 12, 16, 17], [3, 8, 11], [4, 9, 13], [5, 7, 19], [6, 14, 15]]
folds = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19]]
fold_idx = 0
for fold in folds:
    model.load_weights('init_w')
    model.compile(loss=["mean_squared_error"], optimizer=optimizers.Adam(lr))
    patience = 0
    best_loss = np.inf
    epoch = 0
    while True:
        experiment.set_epoch(epoch)

        MAE_training = []
        MAE_testing = []


        # K-fold training
        inp = []
        target = []
        count = 0
        for idx in range(5):#len(gen_train)):
            x, y = gen_train[idx]
            if x[0][0, 0] not in fold:
                shft = (random.randint(-20, 20), random.randint(-20, 20))
                inp.extend(np.expand_dims(np.expand_dims(shift(x[1][0, :, :, 0], shft, cval=0), 0), 3))
                target.extend(np.expand_dims(np.expand_dims(shift(y[0][0, :, :, 0], shft, cval=-1), 0), 3))
                count += 1
            
            if (count == batch_size):
                loss = model.train_on_batch(np.array(inp), np.array(target))
                MAE_training.append(loss)
                inp = []
                target = []
                count = 0
        gen_train.on_epoch_end()
        experiment.log_metrics({"training_loss"+str(fold_idx):np.mean(MAE_training),
                                "fold":fold_idx})
        # K-fold testing
        count = 0
        inp = []
        target = []
        mask = []
        kloss = []
        for k in fold:
            for idx in range(5):#len(gen_val)):
                x, y = gen_val[idx]
                if x[0][0, 0] == k:
                    inp.extend(x[1])
                    target.extend(y[0])
                    mask.extend(y[1])
                    count += 1
                
                if (count == batch_size) | ((idx == (len(gen_val) - 1) & count > 0)):
                    pred = model.predict(np.array(inp))
                    MAE_testing.append(mask_loss(pred, np.array(target), np.array(mask)))
                    inp = []
                    target = []
                    mask = []
                    count = 0
            gen_val.on_epoch_end()
            kloss.append(np.mean(MAE_testing))
        
        experiment.log_metrics({"validation_loss"+str(fold_idx):np.mean(kloss)})
        if (np.mean(best_loss) > np.mean(kloss)):
            best_loss = kloss
            patience = 0
        else:
            patience += 1
        epoch += 1

        if (patience > 0):
            overall.append(best_loss)
            for i in range(3):
                MAE_testing = []
                count = 0
                inp = []
                target = []
                mask = []
                kloss = []
                for k in fold:
                    for idx in range(len(gen_test)):
                        x, y = gen_test[idx]
                        if x[0][0, 0] == k:
                            inp.extend(x[i+1])
                            target.extend(y[0])
                            mask.extend(y[1])
                            count += 1
                        
                        if (count == batch_size) | ((idx == (len(gen_test) - 1) & count > 0)):
                            pred = model.predict(np.array(inp))
                            MAE_testing.append(mask_loss(pred, np.array(target), np.array(mask)))
                            inp = []
                            target = []
                            mask = []
                            count = 0
                gen_test.on_epoch_end()
                experiment.log_metrics({"test_loss_"+str(i):np.mean(MAE_testing)}, step=fold_idx)
                print(np.mean(MAE_testing))
                plot_k = fold[0]
                data = np.zeros((200, 256, 256, 1))
                ct_data = np.zeros((200, 256, 256, 1))
                plot_z = 0
                for idx in range(len(gen_test)):
                    x, y = gen_test[idx]
                    if x[0][0, 0] == plot_k:
                        ct_data[x[4][0, 0], :, :, 0] = y[0][0, :, :, 0]
                        data[x[4][0, 0], :, :, 0] = x[i+1][0, :, :, 0]
                        plot_z += 1
                data = data[0:plot_z, :, :, :]
                ct_data = ct_data[0:plot_z, :, :, :]
                pred = model.predict_on_batch(data)
                plt.figure(figsize=(20, 12))
                plt.subplot(341)
                plt.imshow(data[:, 150, :, 0], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(342)
                plt.imshow(ct_data[:, 150, :, 0], vmin=-1, vmax=1, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(343)
                plt.imshow(pred[:, 150, :, 0], vmin=-1, vmax=1, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(344)
                plt.imshow(pred[:, 150, :, 0] - ct_data[:, 150, :, 0], vmin=-1, vmax=1, cmap='gray')
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.subplot(345)
                plt.imshow(data[:, :, 150, 0], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(346)
                plt.imshow(ct_data[:, :, 150, 0], vmin=-1, vmax=1, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(347)
                plt.imshow(pred[:, :, 150, 0], vmin=-1, vmax=1, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(348)
                plt.imshow(pred[:, :, 150, 0] - ct_data[:, :, 150, 0], vmin=-1, vmax=1, cmap='gray')
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.subplot(349)
                plt.imshow(data[15, :, :, 0], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(3,4,10)
                plt.imshow(ct_data[15, :, :, 0], vmin=-1, vmax=1, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(3,4,11)
                plt.imshow(pred[15, :, :, 0], vmin=-1, vmax=1, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(3,4,12)
                plt.imshow(pred[15, :, :, 0] - ct_data[15, :, :, 0], vmin=-1, vmax=1, cmap='gray')
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                experiment.log_figure(figure=plt, figure_name=exps[i])
                plt.close('all')

            fold_idx += 1
            clear_session()
            gc.collect()
            break
print(folds)
print(overall)
experiment.log_metrics({"overall":np.mean(overall)})
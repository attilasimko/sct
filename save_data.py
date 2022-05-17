import os
import pydicom
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import concatenate

def signal(inp, te, tr):
    return inp[0] * (1 - np.exp(-(tr/1000) / inp[1])) * np.exp(-(te/1000)/ inp[2])

def crop_image(img, thr, defval):
    mask = img >= thr
    mask = ndimage.binary_dilation(mask, iterations=2)
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask = sizes > 512*512*0.08
    mask = mask[label_im]
    mask = ndimage.binary_fill_holes(mask)
    img[~mask] = defval
    return img

data_path = "" # Path to dataset
base_dir =  "" # Path to output data
model_path = "" # Path to contrast transfer model
st1w = False # Create contrast transfer files or not.

training_baseline_patients = ["PELVIS1002MO", "PELVIS1003MP", "PELVIS1004MP", "PELVIS1005MP", "PELVIS1006MR", 
"PELVIS1008MP", "PELVIS1010MR", "PELVIS1011MR", "PELVIS1012MP", "PELVIS1013MO", 
"PELVIS1014MR", "PELVIS1015MP", "PELVIS1016MP", "PELVIS1017MP", "PELVIS1019MP", 
"PELVIS1020MP", "PELVIS1021ML", "PELVIS1022MP", "PELVIS1023MP", "PELVIS1024MP", 
"PELVIS1025MP", "PELVIS1026MP", "PELVIS1027MP", 
"102", "103", "105", "106", "107", "108", "203", "205", "206"]
training_abnormal_patients = ["PELVIS2012MP", "PELVIS2013MP", "PELVIS2016MP", 
                              "PELVIS2712ML", "PELVIS2713ML", "PELVIS2714ML",
                              "PELVIS2814MR", "PELVIS2907MB", "PELVIS2908MB"]
training_female_patients = ["PELVIS1007FRS", "PELVIS1028FR", "PELVIS1047FG", "PELVIS1049FR",
                            "PELVIS2601FG", "PELVIS2602FG", "PELVIS2603FG", "PELVIS2605FG", "PELVIS2609FG", "PELVIS2610FG",
                            "PELVIS2613FG", "PELVIS2614FG", "PELVIS2810FR", "PELVIS2912FB"]

validating_patients = ["PELVIS1031MP", "PELVIS1032MP", "204", "209", "301", "302"]

testing_baseline_patients = ["101", "104", "210", "211", "303", "304", "PELVIS1001MRS", "PELVIS1009MLS", "PELVIS1018MPI", "PELVIS2005MP", "PELVIS2011MP", 
                             "PELVIS2701ML", "PELVIS2704ML", "PELVIS2802MR", "PELVIS2902MB", "PELVIS2903MB", "PELVIS1029FRI", "PELVIS1030FO", "PELVIS2616FG", "PELVIS2617FG", "PELVIS2803FR", "PELVIS2804FR"]
testing_abnormal_patients = []
testing_female_patients = []
testing_t1w_patients = ["101_T1", "104_T1", "210_T1", "211_T1", "303_T1", "304_T1"]

patients = training_baseline_patients + training_abnormal_patients + training_female_patients + \
            validating_patients + \
            testing_baseline_patients + testing_abnormal_patients + testing_female_patients + testing_t1w_patients

if ((len(np.unique(patients)) != 89) | (len(np.unique(patients)) != len(patients))):
    raise Exception("Patient count does not match (" + str(len(np.unique(patients))) + ")")
t1wte = 7
t1wtr = 500

shutil.rmtree(base_dir)
os.mkdir(base_dir)
os.mkdir(base_dir + "testing")
os.mkdir(base_dir + "testing_t1w")
os.mkdir(base_dir + "validating")
os.mkdir(base_dir + "training")
os.mkdir(base_dir + "training_t1w")

model = load_model(model_path, compile=False)
model = Model(model.inputs[0], concatenate(model.outputs[0], model.outputs[1], model.outputs[2]))
model.compile(loss=['mse'])

lst = os.listdir(data_path)
lst.sort()
only_training = False
for st1w in [False, True]: 
    pat_count = 0
    training_patients = training_baseline_patients + training_abnormal_patients + training_female_patients

    for patient in lst:
        save_path = ""
        mult = 1
        unpaired = False
        CT_STACK = []
        MR_STACK = []
        if (training_patients.count(patient) > 0):
            mult = 1
            pat_count += 1
            if (st1w):
                save_path =  base_dir + 'training_t1w/'
            else:
                save_path =  base_dir + 'training/'
                
        elif ((~only_training) & (validating_patients.count(patient) > 0)):
            save_path = base_dir + 'validating/'
        elif ((~only_training) & (testing_baseline_patients.count(patient) > 0)):
            save_path = base_dir + 'testing/'
        elif ((~only_training) & (testing_t1w_patients.count(patient) > 0)):
            save_path = base_dir + 'testing_t1w/'
        else:
            continue
        
        if (not(os.path.isdir(os.path.join(data_path, patient)))):
            print("Patient does not exist (" + str(patient) + ")")
            continue

        for contrast in os.listdir(os.path.join(data_path, patient)):
            if ((contrast == "MR") | (contrast == "CT")):
                # for scan in os.listdir(os.path.join(data_path, dataset, patient, contrast)):
                STACK = []
                for scan_file in os.listdir(os.path.join(data_path, patient, contrast)):
                    data = pydicom.dcmread(os.path.join(data_path, patient, contrast, scan_file))
                    STACK.append(data)
                
                if (contrast == "MR"):
                    MR_STACK = STACK
                elif (contrast == "CT"):
                    CT_STACK = STACK

                if ((len(MR_STACK) > 0) & (len(CT_STACK) > 0)):
                    if (len(MR_STACK) == len(CT_STACK)):
                        CT_STACK = sorted(CT_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                        MR_STACK = sorted(MR_STACK, key=lambda s: float(s.ImagePositionPatient[2]))

                        print(str(patient) + "\t" + str(len(CT_STACK)))
                        for i in range(len(MR_STACK)):
                            mr =  MR_STACK[i].pixel_array / np.mean(MR_STACK[i].pixel_array)
                            mr = cv2.resize(mr, (512, 512))
                            mr = crop_image(mr, 1, 0)
                            if (np.max(mr) == 0): 
                                continue
                            mr =  (mr - np.mean(mr)) / np.std(mr)
                            
                            if (st1w):
                                smr = cv2.resize(mr, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                                smr = smr - np.min(smr)
                                smr = smr / np.max(smr)

                                smr = signal(model.predict_on_batch(np.expand_dims(np.expand_dims(smr, 0), 3)), t1wte, t1wtr)
                                smr = cv2.resize(smr[0, :, :, 0], (512, 512), interpolation=cv2.INTER_LANCZOS4)
                                smr = (smr - np.mean(smr)) / np.std(smr)

                            ct = (CT_STACK[i].RescaleIntercept + CT_STACK[i].RescaleSlope * CT_STACK[i].pixel_array) / 1000
                            ct = np.clip(cv2.resize(ct, (512, 512)), -1, 1)
                            ct = crop_image(ct, -0.2, -1)
                            if (np.max(ct) == -1): 
                                continue

                            for aug_i in range(mult):
                                pat_count += 1
                                ct_i = ct
                                mr_i = mr
                                if (st1w):
                                    smr_i = smr

                                if aug_i != 0:
                                    rand_x = random.randint(-5, 5)
                                    rand_y = random.randint(-5, 5)
                                    if (aug_i % 2 == 1):
                                        ct_i = np.flip(ct_i, 1)
                                        mr_i = np.flip(mr_i, 1)
                                    ct_i = np.roll(ct_i, rand_x, 0)
                                    ct_i = np.roll(ct_i, rand_y, 1)
                                    mr_i = np.roll(mr_i, rand_x, 0)
                                    mr_i = np.roll(mr_i, rand_y, 1)
                                    if (st1w):
                                        smr_i = np.roll(smr_i, rand_x, 0)
                                        smr_i = np.roll(smr_i, rand_y, 1)

                                if (st1w):
                                    np.savez(save_path + str.join("_", (patient + "t1w", str(i), str(aug_i))),
                                            mr=np.array(smr_i, dtype=np.float32),
                                            ct=np.array(ct_i, dtype=np.float32))
                                            
                                np.savez(save_path + str.join("_", (patient, str(i), str(aug_i))),
                                        mr=np.array(mr_i, dtype=np.float32),
                                        ct=np.array(ct_i, dtype=np.float32))
    only_training = True
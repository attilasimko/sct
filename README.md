# Impact of MR training data on the quality of synthetic CT generation
A repository for the thesis work "Impact of MR training data on the quality of synthetic CT generation" by Gustav Jönsson (jonsson.gustav97@gmail.com).

The pre-trained weights used in the evaluation can be found in the repository:
https://drive.google.com/drive/folders/1JVXoMSE0sCHCCWS2qeGsNlaqFMbS7iqp?usp=sharing

The required packages for running the code are stored in the "requirements.txt".

The codes "training_paired.py" and "training_unpaired.py" provided the paths to the dataset train a model for sCT generation using a pixel-wise loss function, and unpaired GAN training, respectively.

The code "evaluate_sct.py" evaluates the trained models on all the available datasets.
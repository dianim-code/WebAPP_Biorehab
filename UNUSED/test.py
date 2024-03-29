# -*- coding: utf-8 -*-
"""Covid-19 DQN Simple Stratified SocioEcon v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z36d6JLNbX8CZP2h1kGqKY65MNGGWzZY
"""

#STRATIFIED VERSION
#ELU invece che ReLU, terminate reward con win/loose, few episodes, done in memoria (come usarlo? reward molto diversi per done e not done)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import random
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pathlib
import os
import shutil

my_path = os.path.dirname(os.path.abspath(__file__))

MAX_RUN = 52
FRESH_TIME = 0.1

#torch.cuda.get_device_name(0)

model_save_name = 'SimpleStratifiedWeightsSocioEcon.pth'
model_save_name_1 ='SimpleStratifiedWeights.pth'
pathMS1 = os.path.join(my_path,model_save_name_1)
path_socioecon = os.path.join(my_path,model_save_name)
sub_path = 'static\\images\\test'
images_dir_test = os.path.join(my_path,sub_path)


if os.path.isdir(images_dir_test):
    shutil.rmtree(images_dir_test)
    os.mkdir(images_dir_test)
else:
    os.mkdir(images_dir_test)


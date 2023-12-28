import torch
import torchvision as tv
import os
import pickle
from torchvision import transforms as T
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Загрузка словаря (vocab.pkl), который содержит отображение слов к их идентификаторам
with open('vocab.pkl', 'rb') as f:
    words = pickle.load(f)
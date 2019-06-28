import cv2
import numpy as np

import os

import sys
sys.path.insert(0, '../')
from datasets.dataset_path import UCF_101_DATA_PATH

cap = cv2.VideoCapture(os.path.join(UCF_101_DATA_PATH, 'IceDancing', 'v_IceDancing_g01_c01.avi'))
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_side_length = 256

fc = 0
ret = True

buf = np.empty((frameCount, output_side_length, output_side_length, 3), np.dtype('uint8'))

save_directory = os.path.join(UCF_101_DATA_PATH, 'IceDancing')

while (fc < frameCount  and ret):
    ret, img = cap.read()
    img = cv2.resize(img, (output_side_length,output_side_length))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buf[fc] = img
    fc += 1
    
print(buf.shape)
    
np.save(os.path.join(save_directory, 'v_IceDancing_g01_c01.npy'), buf)

cap.release()


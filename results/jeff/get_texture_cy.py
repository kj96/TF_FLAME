import cv2
import numpy as np


source_img_fname = './jeff1.png'

source_img = cv2.imread(source_img_fname)

print(type(source_img))
print(source_img.shape)
print(source_img[0][0])

face_xy = []

for i in range(source_img.shape[0]):
    for j in range(source_img.shape[1]):
        if np.count_nonzero(source_img[i][j]) > 0:
            face_xy.append([i,j])

print("face_xy  :   ", len(face_xy))
face_xy = np.array(face_xy)
print("face_xy  :   ", face_xy.shape)
print("face_xy  :   ", face_xy[0])

np.save('face_xy', face_xy)
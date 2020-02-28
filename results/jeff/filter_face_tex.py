import cv2
import numpy as np


source_img_fname = './me.png'


def filter_tex(source_imgf):
    face_xy = np.load('/home/kj/DL/MoYo/Dev/Face3D/github/TF_FLAME/results/jeff/face_xy.npy')
    source_img = cv2.imread(source_imgf)

    face_xy = {str(ij[0])+'#'+str(ij[1]) for ij in face_xy.tolist()}

    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            if str(i)+'#'+str(j) not in face_xy:
                source_img[i][j]= [167, 144,213]
    print("Writing: ", source_imgf)
    cv2.imwrite(source_imgf, source_img)

if __name__ == '__main__':
    filter_tex(source_img_fname)

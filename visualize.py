import numpy as np 
import cv2
import pickle
import argparse

from utils import get_overlapping_features

def draw_keypoints(img_path, keypoints):
    img = cv2.imread(img_path)
    for i in keypoints:
        cv2.circle(img, (int(i[0]), int(i[1])), 1, (0,255,100), 2)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keypoints and feature descriptors.')
    parser.add_argument(
        '--features', type=str, default='sift',
        help='Features to extract: sift, orb, sfop, superpoint, d2net, lift'
    )
    options = parser.parse_args()

    qry_img_path = 'images/ref.ppm'
    trg_img_path = 'images/trg.ppm'

    qry_file_path = 'results/' + options.features + '/ref.ppm_pickle_file'
    trg_file_path = 'results/' + options.features + '/trg.ppm_pickle_file'

    with open(qry_file_path, 'rb') as f:
        qry_file = pickle.load(f)

    with open(trg_file_path, 'rb') as f:
        trg_file = pickle.load(f)
    
    qry_img_name = qry_file[0]
    qry_img_shape = qry_file[1]
    qry_kp = qry_file[2]
    qry_descr = qry_file[3]

    trg_img_name = trg_file[0]
    trg_img_shape = trg_file[1]
    trg_kp = trg_file[2]
    trg_descr = trg_file[3]

    # Plot all the detected keypoints
    qry_img_kp = draw_keypoints(qry_img_path, qry_kp)
    trg_img_kp = draw_keypoints(trg_img_path, trg_kp)
    img2display = np.concatenate((qry_img_kp, trg_img_kp), axis = 1)
    cv2.imshow('window', img2display)
    cv2.waitKey(0)

    # Plot only the keypoints in the common region
    H = np.loadtxt('images/H')
    qry_kp_overlap, qry_descr_overlap, trg_kp_overlap, trg_descr_overlap = get_overlapping_features(qry_kp, qry_descr, qry_img_shape, H, trg_kp, trg_descr, trg_img_shape)
    qry_img_kp_overlap = draw_keypoints(qry_img_path, qry_kp_overlap)
    trg_img_kp_overlap = draw_keypoints(trg_img_path, trg_kp_overlap)
    img2display = np.concatenate((qry_img_kp_overlap, trg_img_kp_overlap), axis = 1)
    cv2.imshow('window', img2display)
    cv2.waitKey(0)

    
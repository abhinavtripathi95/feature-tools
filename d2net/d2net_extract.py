import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from d2net.lib.model_test import D2Net
from d2net.lib.utils import preprocess_image
from d2net.lib.pyramid import process_multiscale

def get_d2net_features(image, max_edge, max_sum_edges, preprocessing, multiscale, model):
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)
    resized_image = image
    if max(resized_image.shape) > max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=preprocessing
    )

    with torch.no_grad():
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    kp_return = keypoints[:,0:2]
    # print (descriptors)
    # print(descriptors.shape)
    return kp_return, descriptors

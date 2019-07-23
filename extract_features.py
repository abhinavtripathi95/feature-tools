import numpy as np 
import cv2
import argparse
import pickle
import os

# GLOBAL VARIABLES
detector_loaded = None
results_dir = None


def extract_features(img_path, features):
    detector = features # i am lazy and i do not want to change var names
    img = cv2.imread(img_path, 0)
    img_shape = img.shape

    if (detector == 'sift') or (detector == 'orb'):
        kp, descr = detector_loaded.detectAndCompute(img, None)
        keypoints = cv2.KeyPoint_convert(kp)
        # print (descr)
        # print (descr.shape)

    elif detector == 'sfop':
        fillprototype(detector_loaded.mymain, ctypes.POINTER(ctypes.c_float), None)
        c_img_path = ctypes.c_char_p(img_path.encode('utf-8'))
        kp = detector_loaded.mymain(c_img_path) 	# Get the float pointer
        array_size = int(kp[0])                     # First element stores size of the entire array
        cc = np.array(np.fromiter(kp, dtype=np.float32, count=array_size))
        fillprototype(detector_loaded.free_mem, None, [ctypes.POINTER(ctypes.c_float)])
        detector_loaded.free_mem(kp)                # Free memory allocated in C
        no_of_kp = int((array_size-1)/2)            # kp = [array_size, x1, y1, x2, y2, ... ]
        keypoints = np.reshape(cc[1:array_size+1], (no_of_kp, 2))
        keypoints = np.around(keypoints, decimals = 3)
        descr = None

    elif detector == 'superpoint':
        #TODO: Replace VideoStreamer by a function that reads a single image
        img_path_superpoint = 'images'
        vs = VideoStreamer(img_path_superpoint, 0, img_shape[0], img_shape[1], 1, '*.ppm')
        if img_path == 'images/ref.ppm':
            img, status = vs.next_frame() 
        else:
            img, status = vs.next_frame()
            img, status = vs.next_frame()
        print(img.shape)
        # cv2.imshow('window',img)
        # cv2.waitKey(0)
        pts, descr, heatmap = detector_loaded.run(img)
        keypoints = pts.T[:,0:2]
        descr = descr.T
        # print (descr)
        print(descr.shape)        

    elif detector == 'd2net':
        # <-- D2Net Default Parameters
        import imageio
        img = imageio.imread(img_path)
        max_edge = 1600
        max_sum_edges = 2800
        preprocessing = 'caffe'
        multiscale = False
        model = detector_loaded
        # Parameters -->
        keypoints, descr = get_d2net_features(img, max_edge, max_sum_edges, preprocessing, multiscale, model)
        # print (descr)
        # print(descr.shape)        

        
    elif detector == 'lift':
        param = detector_loaded[0]
        pathconf = detector_loaded[1]
        new_kp_list = get_lift_kp(param, pathconf, img_path, img_shape)
        kps = get_lift_orientation(param, pathconf, new_kp_list, img_path)
        keypoints, descr = get_lift_features(param, pathconf, kps, img_path)

    return keypoints, descr, img_shape


def save_features(img_name, img_shape, kp, descr):
    cache = []
    cache.append(img_name)
    cache.append(img_shape)
    cache.append(kp)
    cache.append(descr)
    print('----SAVING RESULTS IN PICKLE FILE----')
    file_path = results_dir + '/' + img_name + '_pickle_file'
    with open(file_path, 'wb') as f:
        pickle.dump(cache, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keypoints and feature descriptors.')
    parser.add_argument(
        '--features', type=str, default='sift',
        help='Features to extract: sift, orb, sfop, superpoint, d2net, lift'
    )
    options = parser.parse_args()


    if detector_loaded == None:
        if options.features == 'sift':
            detector_loaded = cv2.xfeatures2d.SIFT_create()

        if options.features == 'orb':
            detector_loaded = cv2.ORB_create()

        if options.features == 'sfop':
            from sfop.mysfop import fillprototype # This is needed for using ctypes pointers
            import ctypes
            detector_loaded = ctypes.cdll.LoadLibrary('sfop/build/src/libsfop.so')
            print("==> Successfully loaded shared library for SFOP")

        if options.features == 'superpoint':
            import torch
            from superpoint.demo_superpoint import SuperPointNet
            from superpoint.demo_superpoint import SuperPointFrontend
            from superpoint.demo_superpoint import VideoStreamer
            print('==> Loading [SuperPointNet] pre-trained network with default parameters: superpoint_v1.pth')
            use_cuda = torch.cuda.is_available()
            detector_loaded = SuperPointFrontend(weights_path='superpoint/superpoint_v1.pth',
                nms_dist=4,
                conf_thresh=0.015,
                nn_thresh=0.7,
                cuda=use_cuda
            )
            print('==> Successfully loaded pre-trained network.')

        if options.features == 'd2net':
            import torch
            from d2net.d2net_extract import get_d2net_features
            from d2net.lib.model_test import D2Net
            print('==> Loading [D2-net] default model: d2_tf.pth')
            cuda_available = torch.cuda.is_available()
            detector_loaded = D2Net(
                model_file='d2net/d2_tf.pth',
                use_relu=True,
                use_cuda=cuda_available
            )
            print('==> Successfully loaded pre-trained network.')

        if options.features == 'lift':
            # Using original implementation of LIFT
            # https://github.com/cvlab-epfl/LIFT
            # detector_loaded = (param, pathconfig)
            from lift.lift_detect import *
            detector_loaded = lift_model()

        results_dir = 'results/' + options.features
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        qry_img_name = 'ref.ppm'
        trg_img_name = 'trg.ppm'

        # Input Variables: a reference image and a target image
        qry_img_path = 'images/' + qry_img_name
        trg_img_path = 'images/' + trg_img_name

        qry_kp, qry_descr, qry_img_shape = extract_features(qry_img_path, options.features)
        trg_kp, trg_descr, trg_img_shape = extract_features(trg_img_path, options.features)

        save_features(qry_img_name, qry_img_shape, qry_kp, qry_descr)
        save_features(trg_img_name, qry_img_shape, trg_kp, trg_descr)

            


import sys
import os
from copy import deepcopy
import h5py
import time
import numpy as np
import cv2
from lift.lib.custom_types import paramGroup, paramStruct, pathConfig

floatX = 'float32'


def lift_model():
    # ------------------------------------------------------------------------
    # Setup and load parameters
    param = paramStruct()
    param.loadParam('lift/models/configs/picc-finetune-nopair.config', verbose=True)
    pathconf = pathConfig()

    # Initialize pathconf structure
    pathconf.setupTrain(param, 0)
    pathconf.result = 'lift/models/picc-best/'

    detector_loaded = []
    detector_loaded.append(param)
    detector_loaded.append(pathconf)
    return detector_loaded



def get_lift_kp(param, pathconf, image_file_name, image_shape):
    from lift.lib.dump_tools import loadh5
    from lift.lib.filter_tools import apply_learned_filter_2_image_no_theano
    from lift.lib.kp_tools import XYZS2kpList, get_XYZS_from_res_list, saveKpListToTxt
    from lift.lib.sift_tools import recomputeOrientation
    from lift.lib.solvers import TestImage

    # ------------------------------------------------------------------------
    # Parameters:
    num_keypoint = 1000 
    # ------------------------------------------------------------------------
    # Run learned network
    start_time = time.clock()
    total_time = 0
    resize_scale = 1.0

    if os.path.exists(image_file_name.replace(
            "image_color", "image_gray"
    )) and "image_color" in image_file_name:
        image_gray = cv2.imread(image_file_name.replace(
            "image_color", "image_gray"
        ), 0)
        image_color = deepcopy(image_gray)
        image_resized = image_gray
        ratio_h = float(image_resized.shape[0]) / float(image_gray.shape[0])
        ratio_w = float(image_resized.shape[1]) / float(image_gray.shape[1])
    else:
        # read the image
        image_color = cv2.imread(image_file_name)
        image_resized = image_color
        ratio_h = float(image_resized.shape[0]) / float(image_color.shape[0])
        ratio_w = float(image_resized.shape[1]) / float(image_color.shape[1])
        image_gray = cv2.cvtColor(
            image_resized,
            cv2.COLOR_BGR2GRAY).astype(floatX)

    assert len(image_gray.shape) == 2

    end_time = time.clock()
    load_prep_time = (end_time - start_time) * 1000.0
    print("Time taken to read and prepare the image is {} ms".format(
        load_prep_time
    ))

    image_height = image_shape[0]
    image_width = image_shape[1]

    # Multiscale Testing
    scl_intv = getattr(param.validation, 'nScaleInterval', 4)
    # min_scale_log2 = 1  # min scale = 2
    # max_scale_log2 = 4  # max scale = 16
    min_scale_log2 = getattr(param.validation, 'min_scale_log2', 1)
    max_scale_log2 = getattr(param.validation, 'max_scale_log2', 4)
    # Test starting with double scale if small image
    min_hw = np.min(image_gray.shape[:2])
    if min_hw <= 1600:
        print("INFO: Testing double scale")
        min_scale_log2 -= 1
    # range of scales to check
    num_division = (max_scale_log2 - min_scale_log2) * (scl_intv + 1) + 1
    scales_to_test = 2**np.linspace(min_scale_log2, max_scale_log2,
                                    num_division)

    # convert scale to image resizes
    resize_to_test = ((float(param.model.nPatchSizeKp - 1) / 2.0) /
                      (param.patch.fRatioScale * scales_to_test))

    # check if resize is valid
    min_hw_after_resize = resize_to_test * np.min(image_gray.shape[:2])
    is_resize_valid = min_hw_after_resize > param.model.nFilterSize + 1

    # if there are invalid scales and resizes
    if not np.prod(is_resize_valid):
        # find first invalid
        first_invalid = np.where(True - is_resize_valid)[0][0]

        # remove scales from testing
        scales_to_test = scales_to_test[:first_invalid]
        resize_to_test = resize_to_test[:first_invalid]

    print('resize to test is {}'.format(resize_to_test))
    print('scales to test is {}'.format(scales_to_test))

    # Run for each scale
    test_res_list = []
    for resize in resize_to_test:

        # Just designate only one scale to bypass resizing. Just a single
        # number is fine, no need for a specific number
        param_cur_scale = deepcopy(param)
        param_cur_scale.patch.fScaleList = [
            1.0
        ]

        # resize according to how we extracted patches when training
        new_height = np.cast['int'](np.round(image_height * resize))
        new_width = np.cast['int'](np.round(image_width * resize))
        start_time = time.clock()
        image = cv2.resize(image_gray, (new_width, new_height))
        end_time = time.clock()
        resize_time = (end_time - start_time) * 1000.0
        print("Time taken to resize image is {}ms".format(
            resize_time
        ))
        total_time += resize_time

        start_time = time.clock()
        sKpNonlinearity = getattr(param.model, 'sKpNonlinearity', 'None')

        test_res = apply_learned_filter_2_image_no_theano(
            image, pathconf.result,
            param.model.bNormalizeInput,
            sKpNonlinearity,
            verbose=True)
        end_time = time.clock()
        compute_time = (end_time - start_time) * 1000.0
        print('Time taken using opencv for image size {} is {}'
                ' milliseconds'.format(image.shape, compute_time))

        total_time += compute_time

        # pad and add to list
        start_time = time.clock()
        test_res_list += [np.pad(test_res,
                                 int((param.model.nFilterSize - 1) / 2),
                                 # mode='edge')]
                                 mode='constant',
                                 constant_values=-np.inf)]
        end_time = time.clock()
        pad_time = (end_time - start_time) * 1000.0
        print("Time taken for padding and stacking is {} ms".format(
            pad_time
        ))
        total_time += pad_time

    # ------------------------------------------------------------------------
    # Non-max suppresion and draw.

    # The nonmax suppression implemented here is very very slow. COnsider this
    # as just a proof of concept implementation as of now.

    # Standard nearby : nonmax will check the approx. the same area as
    # descriptor support region.
    nearby = int(np.round(
        (0.5 * (param.model.nPatchSizeKp - 1.0) *
         float(param.model.nDescInputSize) /
         float(param.patch.nPatchSize))
    ))
    fNearbyRatio = getattr(param.validation, 'fNearbyRatio', 1.0)
    # Multiply by quarter to compensate
    fNearbyRatio *= 0.25
    nearby = int(np.round(nearby * fNearbyRatio))
    nearby = max(nearby, 1)

    nms_intv = getattr(param.validation, 'nNMSInterval', 2)
    edge_th = getattr(param.validation, 'fEdgeThreshold', 10)
    do_interpolation = getattr(param.validation, 'bInterpolate', True)

    print("Performing NMS")
    fScaleEdgeness = getattr(param.validation, 'fScaleEdgeness', 0)
    start_time = time.clock()
    res_list = test_res_list
    XYZS = get_XYZS_from_res_list(res_list, resize_to_test,
                                  scales_to_test, nearby, edge_th,
                                  scl_intv, nms_intv, do_interpolation,
                                  fScaleEdgeness)

    end_time = time.clock()
    XYZS = XYZS[:num_keypoint]
    print('===========================',XYZS)
    print('===========================',XYZS.shape)
    print('===========================',max(XYZS[:,0]))
    print('===========================',max(XYZS[:,1]))



    nms_time = (end_time - start_time) * 1000.0
    print("NMS time is {} ms".format(nms_time))
    total_time += nms_time
    print("Total time for detection is {} ms".format(total_time))
    

    # # ------------------------------------------------------------------------
    # # Save as keypoint file to be used by the oxford thing
    print("Turning into kp_list")
    kp_list = XYZS2kpList(XYZS)  # note that this is already sorted

    # # ------------------------------------------------------------------------
    # # Also compute angles with the SIFT method, since the keypoint component
    # # alone has no orientations.
    # print("Recomputing Orientations")
    new_kp_list, _ = recomputeOrientation(image_gray, kp_list,
                                          bSingleOrientation=True)

    # print(new_kp_list[0])
    return new_kp_list

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################




def get_lift_orientation(param, pathconf, new_kp_list, image_file_name):
    from six.moves import xrange
    from lift.lib.dataset_tools import test as data_module
    from lift.lib.dump_tools import loadh5
    from lift.lib.kp_tools import IDX_ANGLE, saveKpListToTxt, update_affine
    from lift.lib.solvers import Test

    # ------------------------------------------------------------------------
    # Setup and load parameters
    param = paramStruct()
    param.loadParam('lift/models/configs/picc-finetune-nopair.config', verbose=True)
    pathconf = pathConfig()
    pathconf.setupTrain(param, 0)

    pathconf.result = 'lift/models/picc-best/'
    # Modify the network so that we bypass the keypoint part and the
    # descriptor part.
    param.model.sDetector = 'bypass'
    # This ensures that you don't create unecessary scale space
    param.model.fScaleList = np.array([1.0])
    param.patch.fMaxScale = np.max(param.model.fScaleList)
    # this ensures that you don't over eliminate features at boundaries
    param.model.nPatchSize = int(np.round(param.model.nDescInputSize) *
                                np.sqrt(2))
    param.patch.fRatioScale = (float(param.patch.nPatchSize) /
                            float(param.model.nDescInputSize)) * 6.0
    param.model.sDescriptor = 'bypass'

    # add the mean and std of the learned model to the param
    mean_std_file = pathconf.result + 'mean_std.h5'
    mean_std_dict = loadh5(mean_std_file)
    param.online = paramGroup()
    setattr(param.online, 'mean_x', mean_std_dict['mean_x'])
    setattr(param.online, 'std_x', mean_std_dict['std_x'])

    # -------------------------------------------------------------------------
    # Let's check if we can replace kp_file_name by the actual contents of the file
    # Load data in the test format

    kp_list = new_kp_list
    test_data_in = data_module.data_obj(param, image_file_name, kp_list)

    # -------------------------------------------------------------------------
    # Test using the test function
    start_time = time.clock()
    _, oris, compile_time = Test(
        pathconf, param, test_data_in, test_mode="ori")
    end_time = time.clock()
    compute_time = (end_time - start_time) * 1000.0 - compile_time
    print("Time taken to compile is {} ms".format(
        compile_time
    ))
    print("Time taken to compute is {} ms".format(
        compute_time
    ))
    bPrintTime = False 
    # okay, this was a optional parameter for running compute_orientations
    #  and honestly, i feel this program prints a shitload of stuff already
    if bPrintTime:
        # Also print to a file by appending
        with open("../timing-code/timing.txt", "a") as timing_file:
            print("------ Orientation Timing ------\n"
                    "Computation time for {} keypoints is {} ms\n".format(
                        test_data_in.x.shape[0],
                        compute_time
                    ),
                    file=timing_file)

    # update keypoints and save as new
    kps = test_data_in.coords
    for idxkp in xrange(len(kps)):
        kps[idxkp][IDX_ANGLE] = oris[idxkp] * 180.0 / np.pi % 360.0
        kps[idxkp] = update_affine(kps[idxkp])

    # save as new keypoints
    # we have new keypoints as kps
    # we are not saving them, yet
    # saveKpListToTxt(kps, kp_file_name, output_file)
    print('========2222222222=======',kps[0].shape) #(13,)
    print('========2222222222=======',len(kps[0]))  #13
    print('========2222222222=======',kps.shape)    #(965,13)

    return kps


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################




def get_lift_features(param, pathconf, kps, image_file_name):
    from lift.lib.dataset_tools import test as data_module
    from lift.lib.dump_tools import loadh5, saveh5
    from lift.lib.solvers import Test

    bPrintTime = False
    bSavePng = False
    bDumpPatch = False

    param = paramStruct()
    param.loadParam('lift/models/configs/picc-finetune-nopair.config', verbose=True)
    pathconf = pathConfig()
    pathconf.setupTrain(param, 0)


    pathconf.result = 'lift/models/picc-best/'
    cwd = os.getcwd()
    # Overwrite with hard-coded base model
    setattr(param.model, "descriptor_export_folder",
                cwd  + "/lift/models/base")

    param.model.sDetector = 'bypass'
    # This ensures that you don't create unecessary scale space
    param.model.fScaleList = np.array([1.0])
    param.patch.fMaxScale = np.max(param.model.fScaleList)
    # this ensures that you don't over eliminate features at boundaries
    param.model.nPatchSize = int(np.round(param.model.nDescInputSize) *
                                    np.sqrt(2))
    param.patch.fRatioScale = (float(param.patch.nPatchSize) /
                                float(param.model.nDescInputSize)) * 6.0
    param.model.sOrientation = 'bypass'

    # add the mean and std of the learned model to the param
    mean_std_file = pathconf.result + 'mean_std.h5'
    mean_std_dict = loadh5(mean_std_file)
    param.online = paramGroup()
    setattr(param.online, 'mean_x', mean_std_dict['mean_x'])
    setattr(param.online, 'std_x', mean_std_dict['std_x'])

    # -------------------------------------------------------------------------

    # Load data in the test format
    # test_data_in expects a list of kps, not an np array
    kp_list = kps.tolist()
    print('kp_list_len', len(kp_list))
    # print('============33333333333==============', kp_list)
    test_data_in = data_module.data_obj(param, image_file_name, kp_list)

    # -------------------------------------------------------------------------
    # Test using the test function
    start_time = time.clock()
    descs, _, compile_time = Test(
        pathconf, param, test_data_in, test_mode="desc")
    end_time = time.clock()
    # print('========================33333333333===========3', descs)
    compute_time = (end_time - start_time) * 1000.0 - compile_time
    print("Time taken to compile is {} ms".format(
        compile_time
    ))
    print("Time taken to compute is {} ms".format(
        compute_time
    ))
    if bPrintTime:
        # Also print to a file by appending
        with open("../timing-code/timing.txt", "a") as timing_file:
            print("------ Descriptor Timing ------\n"
                    "Computation time for {} keypoints is {} ms\n".format(
                        test_data_in.x.shape[0],
                        compute_time
                    ),
                    file=timing_file)

    save_dict = {}
    save_dict['keypoints'] = test_data_in.coords
    save_dict['descriptors'] = descs
    print(test_data_in.coords)
    print(test_data_in.coords.shape)
    print(descs.shape)
    descr = descs.astype(np.float32)
    keypoints2return = test_data_in.coords[:,0:2]
    return keypoints2return, descr
import numpy as np

def get_overlapping_features(qry_kp, qry_descr, qry_img_shape, H, trg_kp, trg_descr, trg_img_shape):
    Hinv = np.linalg.inv(H)

    ############ Overlay the images to find the ##############
    ######### keypoints in the overlapping region ############
    homogenous_qry = np.transpose(np.c_[qry_kp, np.ones(len(qry_kp))])
    hx = np.dot(H, homogenous_qry)
    z = hx[-1,:]
    hx = hx/z[None,:]
    # Convert homogenous coordinates into non-homogenous
    hx = hx[0:2,:]
    # print(qry_kp.shape)
    # print(hx.shape)
    qry_kp_original = qry_kp[(hx[0] >= 0) & (hx[0] <= trg_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= trg_img_shape[0]), :]
    qry_descr_original = qry_descr[(hx[0] >= 0) & (hx[0] <= trg_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= trg_img_shape[0]), :]
    qry_kp = hx[:, (hx[0] >= 0) & (hx[0] <= trg_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= trg_img_shape[0])] 
    qry_kp = np.transpose(qry_kp)

    homogenous_trg = np.transpose(np.c_[trg_kp, np.ones(len(trg_kp))])
    hx = np.dot(Hinv, homogenous_trg)            
    z = hx[-1,:]
    hx = hx/z[None,:]
    # Convert homogenous coordinates into non-homogenous
    hx = hx[0:2,:]
    trg_kp = trg_kp[(hx[0] >= 0) & (hx[0] <= qry_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= qry_img_shape[0]), :]
    trg_descr = trg_descr[(hx[0] >= 0) & (hx[0] <= qry_img_shape[1]) & 
                        (hx[1] >= 0) & (hx[1] <= qry_img_shape[0]), :]
    # point_qry_len = len(qry_kp)
    # point_trg_len = len(trg_kp)
    # print('    Overlapping kp in query image: ', point_qry_len)
    # print('    Overlapping kp in target image: ', point_trg_len)
    return qry_kp_original, qry_descr_original, trg_kp, trg_descr
    # NOTE: qry_kp stores the keypoints overlayed on the target image,
    # however qry_kp_original stores the coordinates on the original
    # query image
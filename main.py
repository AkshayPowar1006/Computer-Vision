import os, math
import time
from cv2 import findHomography
import numpy as np
import cv2 as cv
import scipy.io as io
import sklearn
from sklearn.feature_extraction import image

# Define settings for your experiment

settings = {}
settings['dataset'] = 'carla' # choose from ['kitti', 'carla']

settings["patch_size"] = 7 #7
if settings['dataset']=='kitti':
    settings["data_path"] = './data/kitti'
else:
    settings["data_path"] = './data/carla'

settings["results_directory"] = './results/' + settings['dataset']

# We should down size the images, to see results quickly
settings["width"] = 800#512
settings["height"] = 600#256


# Num of depth proposals
settings["num_depths"] = 100
settings["min_depth"] =  2.0 # in meters
settings["max_depth"] =  20000.0

settings["similarity"] = "SSD"
os.makedirs(settings["results_directory"], exist_ok=True)

def get_depth_proposals(min_depth, max_depth, num_depths):
    '''
    return list of depth proposals
    you can sample the range [min_depth, max_depth] uniformly at num_depths points.
    Tip: linearly sampling depth range doesnot lead to a linear step along the epipolar line.
    Instead, linearly sample the inverse-depth [1/min_depth, 1/max_depth] then take its inverse to get depth values.
    This is practically more meaningful as it leads to linear step in pixel space.
    '''
    depth_proposals_inv = np.linspace(1/min_depth, 1/max_depth, num_depths)
    return 1/depth_proposals_inv

def depth_to_file(depth_map, filename):
    """
    Saves depth maps to as images
    feel free to modify it, it you want to get fancy pics
    """
    depth_ = 1/(depth_map+0.00001)
    depth_ = 255.0*depth_/(np.percentile(depth_.max(), 95))
    cv.imwrite(filename, depth_)

def copy_make_border(img, patch_width):
    """
    This function applies cv.copyMakeBorder to extend the image by patch_width/2
    in top, bottom, left and right part of the image
    Patches/windows centered at the border of the image need additional padding of size patch_width/2
    """
    offset = np.int(patch_width/2.0)
    return cv.copyMakeBorder(img,
                             top=offset, bottom=offset,
                             left=offset, right=offset,
                             borderType=cv.BORDER_REFLECT)

def extract_pathches(img, patch_width):
    '''
    Input:
        image: size[h,w,3]
    Return:
        patches: size[h, w, patch_width, patch_width, c]
    '''
    if img.ndim==3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1
    img_padded = copy_make_border(img, patch_width)
    patches = image.extract_patches_2d(img_padded, (patch_width, patch_width))
    patches = patches.reshape(h, w, patch_width, patch_width, c)
    return patches

def read_kitti_calib_file():
    filename = os.path.join(settings["data_path"], 'calib.txt')
    data = np.fromfile(filename, sep=' ').reshape(3,4)[0:3,0:3]
    return data

def read_carla_calib_file():
    fov=90.0
    height=600
    width=800
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / \
                        (2.0 * math.tan(fov * math.pi / 360.0))
    return k

def load_imgs_and_k_mats():
    img_0 = cv.imread(os.path.join(settings['data_path'], 'images', '0.png'))
    img_h, img_w, c = img_0.shape
    # Load and Downsize the images, for faster computation
    height, width = settings['height'], settings['width']
    imgs = [cv.resize(cv.imread(os.path.join(settings["data_path"], 'images', str(ii)+'.png')),\
    (settings['width'], settings['height']))\
    for ii in range(5)]
    source_img = imgs.pop(2)
    input_imgs = imgs
    if settings['dataset']=='kitti':
        k_matrix = read_kitti_calib_file()
    else:
        k_matrix = read_carla_calib_file()
    k_matrix[0,:] = k_matrix[0,:]*float(width)/float(img_w)
    k_matrix[1,:] = k_matrix[1,:]*float(height)/float(img_h)
    return input_imgs, source_img, k_matrix

def load_camera_pose():
    if settings['dataset']=='kitti':
        filename = os.path.join(settings["data_path"], 'cam_pose.txt')
        data = np.fromfile(filename, sep=' ').reshape(5, 3,4)
        RMats = data[:,0:3,0:3]
        TVecs = data[:,:,3]
        # We should make the middle view as our source view.
        mid = 2
        ref_R = RMats[mid]
        ref_T = TVecs[mid]
        rot_mat_list = []
        t_vec_list = []
        for ii in range(5):
            R, T = RMats[ii], TVecs[ii]
            R_ii = np.dot(np.linalg.inv(R), ref_R)
            T_ii = np.dot(np.linalg.inv(R), (ref_T-T)).reshape(3,1)
            rot_mat_list.append(R_ii)
            t_vec_list.append(T_ii)
    else:
        # This is for carla dataset
        # do not change this function
        t_vec_list = [np.array([0, 0, (i-2)*0.54], dtype=np.float).reshape(3,1) for i in range(5)]
        rot_mat_list = [np.eye(3) for i in range(5)]
    return rot_mat_list, t_vec_list


# Load Images, K-Matrix and Camera Pose
# Except K matric which is 3x3 array, other parameters are lists

input_imgs, source_img, k_matrix = load_imgs_and_k_mats()
# input_imgs is a list of 4 images while source_img is a single image
r_matrices, t_vectors = load_camera_pose()
# r_matrices and t_vectors are lists of length 5
# values at index 2 are for the middle camera which is the reference camera
# print(r_matrices[2]) # identity
# print(t_vectors[2]) # zeros
# print(k_matrix) # a 3x3 matrix

def ssd(feature_1, feature_2):      #from solution exercise 4, modified
    '''
    inputs: feature_1 and feature_2 with shape[w, h, p, p , 3]
    p = patch_width
    return: per pixel distance , shape[w, h]
    '''
    sq_diff = np.square(feature_1 - feature_2).astype('float')
    ssd_val = np.sum(sq_diff, axis=(2,3,4))
    return ssd_val

def mean_ssd(feature_1, feature_2, feature_3, feature_4):
    ssd_val12 = ssd(feature_1, feature_2)
    ssd_val13 = ssd(feature_1, feature_3)
    ssd_val14 = ssd(feature_1, feature_4)
    ssd_val23 = ssd(feature_2, feature_3)
    ssd_val24 = ssd(feature_2, feature_4)
    ssd_val34 = ssd(feature_3, feature_4)
    return (ssd_val12 + ssd_val13 + ssd_val14 + ssd_val23 + ssd_val24 + ssd_val34)/6

def computeHomography(depth, K_inv, P):
    '''
    inputs:
        depth: depth_value (float)
        K_inv : Inverses of the camera matrix
        P: Projection matrices of the target view
    return: Homography, that maps Points with depth depth from the source view to the corresponding possition of the
        input view
    '''
    Corners_hom = np.array([[0,0,1], [settings['height'],0,1], [0,settings['width'],1], [settings['height'],settings['width'],1]])
    Corners_3D_depth = depth * np.einsum('pi,ji', Corners_hom, K_inv)
    Corners_3d_depth_hom = np.vstack((Corners_3D_depth, np.ones(4))).T
    Projec_hom = np.einsum('pi,ji', Corners_3d_depth_hom, P).T
    Projec = np.vstack(((Projec_hom.T[0]/Projec_hom.T[2]),(Projec_hom.T[1]/Projec_hom.T[2]))).T
    Hom, m = cv.findHomography(Corners_hom[:,:2],Projec)
    return Hom

def ssd_merge(ssd_values):
    '''
    input: ssd_ values: float array containing for each pixel the ssd values for allinput views, shape = (h,w,4)
    output: array of for each pixel combined ssd values. shape = (h,w) 
    '''
    return np.mean(ssd_values, axis = 2)

if __name__ == '__main__':
    patch_size = settings["patch_size"]
    num_depths = settings["num_depths"]
    min_depth = settings["min_depth"]
    max_depth = settings["max_depth"]
    Depth_proposals = get_depth_proposals(settings['min_depth'], settings['max_depth'], num_depths=settings['num_depths'])
    Patches_source = extract_pathches(source_img, settings['patch_size'])

    #Extract Patches
    Patches_source = extract_pathches(source_img,settings['patch_size'])
    Patches_input = []
    for im in input_imgs:
        Patches_input.append(extract_pathches(im, patch_size))
    Patches_input = np.array(Patches_input)     #has shape (#input cams, width, height)

    # Compute projection matrices
    P_matrices = []
    for i in [0,1,3,4]:
        P_mat = k_matrix @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) @ np.vstack((np.hstack((r_matrices[i], t_vectors[i])), np.array([0,0,0,1])))
        P_matrices.append(P_mat)
    P_matrices = np.array(P_matrices)
    K_inv = np.linalg.inv(k_matrix) 

    # ___________________________________________Task 1___________________________________________
    max_ssd = -1 * np.ones((settings['height'], settings['width']))
    Depth_map = np.zeros((settings['height'], settings['width']))
    for d in Depth_proposals:
        ssd_Scores = np.zeros((settings['height'], settings['width'],4))
        for c in range(4):
            Input_img = input_imgs[c]
            Hom = computeHomography(d, K_inv, P_matrices[c])
            WarpedInput = cv.warpPerspective(Input_img, Hom, (settings['width'],settings['height']))
            Patches_WarpedInput = extract_pathches(WarpedInput, settings['patch_size'])
            ssd_Scores_c = ssd(Patches_source, Patches_WarpedInput)
            ssd_Scores[:,:,c] = ssd_Scores_c
        ssd_Scores_comb = ssd_merge(ssd_Scores)
        ssd_greater = np.greater(ssd_Scores_comb,max_ssd)
        max_ssd = np.where(ssd_greater, ssd_Scores_comb, max_ssd)
        Depth_map = np.where(ssd_greater, d*np.ones_like(Depth_map), Depth_map)
        depth_to_file(Depth_map, './results/'+ settings['dataset'] +'/depthmap'+ \
            str(settings["patch_size"]) + 'x' + str(settings["patch_size"]) + '.jpg')

    # ___________________________________________Task 2___________________________________________
    ReconstructCam2 = np.zeros_like(source_img)
    ssd_opt = -1*np.ones((settings['height'],settings['width']))
    for d in Depth_proposals:
        Hom0 = computeHomography(d, K_inv, P_matrices[0])
        Hom1 = computeHomography(d, K_inv, P_matrices[1])
        Hom3 = computeHomography(d, K_inv, P_matrices[2])
        Hom4 = computeHomography(d, K_inv, P_matrices[3])
        WarpedCam0 = cv.warpPerspective(input_imgs[0], Hom0, (settings['width'],settings['height']))
        WarpedCam1 = cv.warpPerspective(input_imgs[1], Hom1, (settings['width'],settings['height']))
        WarpedCam3 = cv.warpPerspective(input_imgs[2], Hom3, (settings['width'],settings['height']))
        WarpedCam4 = cv.warpPerspective(input_imgs[3], Hom4, (settings['width'],settings['height']))
        Patches_WarpedCam0 = extract_pathches(WarpedCam0, settings['patch_size'])
        Patches_WarpedCam1 = extract_pathches(WarpedCam1, settings['patch_size'])
        Patches_WarpedCam3 = extract_pathches(WarpedCam3, settings['patch_size'])
        Patches_WarpedCam4 = extract_pathches(WarpedCam4, settings['patch_size'])
        ssd_Scores = mean_ssd(Patches_WarpedCam0,Patches_WarpedCam1,Patches_WarpedCam3,Patches_WarpedCam4)
        warp_combined = np.mean(np.array([WarpedCam0, WarpedCam1, WarpedCam3, WarpedCam4]), axis = 0)       #choose between mean and median
        ssd_greater = np.greater(ssd_Scores,ssd_opt)
        ssd_opt = np.where(ssd_greater, ssd_Scores, ssd_opt)
        ReconstructCam2[:,:,0] = np.where(ssd_greater, warp_combined[:,:,0], ReconstructCam2[:,:,0])
        ReconstructCam2[:,:,1] = np.where(ssd_greater, warp_combined[:,:,1], ReconstructCam2[:,:,1])
        ReconstructCam2[:,:,2] = np.where(ssd_greater, warp_combined[:,:,2], ReconstructCam2[:,:,2])
        cv.imwrite('./results/'+ settings['dataset'] + '/synthesis'+ \
               str(settings["patch_size"]) + 'x' + str(settings["patch_size"]) + 'mean' + '.jpg', ReconstructCam2)
    


    print("Done")
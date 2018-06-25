import numpy as np
import cv2

from skimage import filters
from skimage.feature import local_binary_pattern
from skimage.morphology import dilation, opening, closing
from skimage.morphology import disk
from scipy import ndimage

from custom_image_tool import normalize


# https://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white
def old_opencv_threshold(img, thresh=None, max_value=255):
    if not img.dtype == np.uint8:
        img = normalize(img)
    if thresh is not None:
        return (cv2.threshold(img, thresh, max_value, cv2.THRESH_BINARY)[1]).astype(np.uint8)
    else:
        return (cv2.threshold(img, 0, 255, cv2. THRESH_BINARY +cv2.THRESH_OTSU)[1]).astype(np.uint8)


def opencv_th_adaptive_mean(img, max_value=255, block_size=11):
    if not img.dtype == np.uint8:
        img = normalize(img)
        return (cv2.adaptiveThreshold(img, max_value, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, blockSize=block_size, C=2)).astype(np.uint8)
    else: return None


def opencv_th_adaptive_gaussian(img, max_value=255, block_size=11):
    if not img.dtype == np.uint8:
        img = normalize(img)
        return (cv2.adaptiveThreshold(img, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, blockSize=block_size, C=2)).astype(np.uint8)
    else: return None


# Otsu's thresholding
def opencv_th_otsu(img, max_value=255):
    if not img.dtype == np.uint8:
        img = normalize(img)
    return (cv2.threshold(img, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]).astype(np.uint8)


# Otsu's thresholding after Gaussian filtering
def opencv_th_otsu_after_gauss_blur(img, max_value=255, block_size=(5,5), sigmaX=0):
    if not img.dtype == np.uint8:
        img = normalize(img)
        _blur = cv2.GaussianBlur(img, block_size, sigmaX)
        return (cv2.threshold(_blur, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]).astype(np.uint8)
    else: return None


def skimage_threshold(img, method='isodata'):
    if not img.dtype == np.uint8:
        print('skimage_threshold NORM because received: ', img.dtype)
        img = normalize(img)

    thresh = {
        'otsu': filters.threshold_otsu(img), # delivers very high threshold
        'yen': filters.threshold_yen(img), # delivers even higher threshold
        'isodata' :filters.threshold_isodata(img), # works extremely well
    }.get(method, 0)  # None is default value
    if thresh == 0: print(' ***  ERROR - skimage_threshold select THRESHOLD = 0  ***')
    return thresh, cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]


def opencv_th_k_means_th(img, K=2, max_iter=10, epsilon=1.0, n_ripet=10):
    img = normalize(img)
    feat_ldg = img.reshape((-1, 1))
    label, center = k_means_clustering(feat_ldg, K, max_iter, epsilon, n_ripet)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    return np.array([res.reshape(img.shape), np.array(center), label])


def k_means_clustering(features, K, max_iter=10, epsilon=1.0, n_ripet=10):
    """ features -> matrix with each feature in a single column.
    """
    if np.size(features.shape) != 2:
        raise ValueError('features has wrong dimension: {}'.format(features.shape))
    else:
        features = np.float32(features)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
        ret, label, center = cv2.kmeans(features, K, None, criteria, n_ripet, cv2.KMEANS_RANDOM_CENTERS)
        return label, center


def make_cells_mask(center, label, img_shape, K=4, true=2):
    # TODO controllo sulla lunghezza di center (se non è 4?)
    # idx_sorted have length K:
    # melts label from 0 -> K-true to 0 (low)   ,  and label from true->K (high) to 255
    idx_sorted = np.argsort(center.flatten())
    center[idx_sorted[0: K - true]] = 0
    center[idx_sorted[K - true: K]] = 255
    mask = center[label.flatten()]
    return mask.reshape(img_shape)


def widens_mask(img):
    """ Morphological operators on img """
    img = closing(img, disk(2))
    img = opening(img, disk(1))
    img = remove_little_holes(img, destructive=False)  # remove little black holes
    neg = ~img
    img = ~remove_little_holes(neg, destructive=False)  # remove little white elements
    return dilation(img, disk(1))


def widens_mask_deconv(mask, t_ratio_holes, _plot=False):
    tot_pixel = np.prod(mask.shape)
    max_size = int((t_ratio_holes * tot_pixel) / 100)
    
    mask = closing(mask, disk(3))
    mask = remove_little_holes(mask, min_size=0, max_size=max_size)  # remove little black holes
    neg = ~mask
    mask = ~remove_little_holes(neg, max_size=max_size)  # remove little white elements
    mask = opening(mask, disk(3))
    mask = dilation(mask, disk(1))
    return mask


def remove_little_holes(img, min_size=0, max_size=100, destructive=False):
    """ destructive is important!! if TRUE, this function modifies the input image"""
    invert_im = np.where(img == 0, 1, 0)
    label_im, num = ndimage.label(invert_im)
    holes = ndimage.find_objects(label_im)
    small_holes = [hole for hole in holes if min_size < img[hole].size <= max_size]

    if destructive is True:
        filled_image = img  # filled_image
    else:
        filled_image = np.copy(img)  # duplicate and work on it

    for hole in small_holes:
        a, b, c, d = (max(hole[0].start - 1, 0),
                      min(hole[0].stop + 1, img.shape[0] - 1),
                      max(hole[1].start - 1, 0),
                      min(hole[1].stop + 1, img.shape[1] - 1))

        filled_image[a:b, c:d] = ndimage.morphology.binary_fill_holes(filled_image[a:b, c:d]).astype(np.uint8) * 255
    return filled_image


def th_ldg_texture_kmeans(img, K=2, lbp_angular_res=16, lbp_spatial_res=16, lbp_method='ror'):

    # extract texture using local binary pattern
    img_texture = local_binary_pattern(img, P=lbp_angular_res, R=lbp_spatial_res, method=lbp_method)
    # arrays of features
    ldg = (img.reshape((-1, 1))).astype('float32')
    tex = (img_texture.reshape((-1, 1))).astype('float32')
    # normalize to [0,1] and insert in two column
    ldg *= (1.0 / ldg.max())
    tex *= (1.0 / tex.max())

    features = (np.column_stack((ldg, tex)))

    # k-means clustering on features
    label, centers = k_means_clustering(features=features, K=K)

    # Now convert ldg centroid back into uint8, and make original image
    # NOTE - here centers have 2 column (ldg, tex), ...
    # i want recreate image with ldg values for every cluster
    centers_ldg = np.uint8(255*centers[:, 0])
    res = centers_ldg[label.flatten()]
    return np.array([res.reshape(img.shape), np.array(centers_ldg), label])

import numpy as np
import os

from skimage.external.tifffile import imsave

from custom_tool_kit import magnitude


# TODO: espamdere ai tre canali con ...
def bleach_correction(vol, z_ref=None, verbose=False):
    '''
    Simple Ratio Bleah correction in the 3d data 'vol'
    '''
    print(vol.shape) # remove

    if vol is not None:
        if len(vol.shape) == 3:
            if vol.shape[0] > 1:

                if z_ref is None:
                    # extract middle frame intensity
                    z_ref = int(vol.shape[0] / 2)
                    while True:
                        ref_value = np.mean(vol[z_ref])
                        if ref_value != 0:
                            break
                        else:
                            z_ref = z_ref + 1
                else:
                    ref_value = np.mean(vol[z_ref])

                # bleach correction
                for z in range(vol.shape[0]):
                    if verbose: print('z: {}'.format(z), end='')
                    m = np.mean(vol[z])
                    if m != 0:
                        ratio = ref_value / m
                        vol[z] = vol[z] * ratio
                        if verbose: print(' - ratio: {0:0.3f}'.format(ratio))
                    else:
                        if verbose: print(' - is empty')
    return vol


def save_tiff(img, img_name, comment='', folder_path='', prefix=''):
    # folder_path must end with '/'

    # check if name end with .tif
    if img_name.endswith('.tiff'):
        base, ext = os.path.splitext(img_name)
        img_name = prefix + base + '_' + comment + '.tiff'
    else:
        img_name = prefix + img_name + '_' + comment + '.tiff'

    # check if path end with '/'
    if not folder_path.endswith('/'):
        folder_path += '/'

    # imsave(folder_path + img_name, (img * (255 / np.max(img))).astype(np.uint8))
    imsave(folder_path + img_name, img)


def image_have_info(img, t=0.09):
    rating = np.sum(img) / np.prod(img.shape)
    if rating > t:
        return True
    else:
        return False


def normalize(img, max_value=255.0, dtype=np.uint8):
    max_v = img.max()
    min_v = img.min()
    if max_v != 0:
        if max_v != min_v:
            return (((img - min_v) / (max_v - min_v)) * max_value).astype(dtype)
        else:
            return ((img / max_v) * max_value).astype(dtype)
    else:
        return img.astype(dtype)


def normalize_ext(img, max_value=None, dtype=None):
    """ Stretch or shrink the values of img.
        img is a numpy array """
    if img is not None:
        if dtype is None:
            dtype = img.dtype
        if max_value is None:
            try:
                max_value = np.iinfo(dtype).max
            except (NameError, ValueError):
                pass

            try:
                max_value = np.finfo(dtype).max
            except (NameError, ValueError):
                pass

            finally:
                max_value = 255 if max_value is None else max_value

        max_v = img.max()
        min_v = img.min()
        if max_v != 0:
            if max_v != min_v:
                return (((img - min_v)/(max_v - min_v)) * max_value).astype(dtype)
            else:
                return ((img / max_v) * max_value).astype(dtype)
        else:
            return img.astype(dtype)
    return None


def split_channel(img):
    img_red = np.array(img)[:, :, 0]
    img_green = np.array(img)[:, :, 1]
    img_blue = np.array(img)[:, :, 2]
    return img_red, img_green, img_blue


def print_info(X, text=''):
    if X is None:
        return None
    print(text)
    print(' * Image dtype: {}'.format(X.dtype))
    print(' * Image shape: {}'.format(X.shape))
    print(' * Image max value: {}'.format(X.max()))
    print(' * Image min value: {}'.format(X.min()))
    print(' * Image mean value: {}'.format(X.mean()))


def create_img_name_from_index(i, pre='', post='', tot_digits=5):
    if 0 <= i < 10 ** tot_digits - 1:
        if i == 0:
            non_zero_digits = 1
        else:
            non_zero_digits = magnitude(i) + 1
        zeros = '0' * (tot_digits - non_zero_digits)
        return str(pre + zeros + str(i) + post + '.tif')
    else:
        return 'image_index_out_of_range'

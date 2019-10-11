import numpy as np
import matplotlib.pyplot as plt

import pymrt.geometry as geom
from scipy.ndimage.measurements import center_of_mass


def fft(img):
    """
    :param img:
    :return: spectrum of img"""
    if img is not None:
        f = np.fft.fft2(img)  # fft 
        return np.fft.fftshift(f)  # shift for centering 0.0 (x,y)
    else:
        raise ValueError('img is None')


def ifft(spectrum):
    if spectrum is not None:
        f_ishift= np.fft.ifftshift(spectrum)  # shift for centering 0.0 (x,y)
        img_back = np.fft.ifft2(f_ishift)  # ifft
        return np.abs(img_back)
    else:
        raise ValueError('spectrum is None')


def plot_spectrum(spectrum, title='', _grid=False):
    if spectrum is not None:
        plt.figure(figsize=(14, 14))
        magnitude_spectrum = 20 * np.log(np.abs(spectrum))
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title(title)
        plt.grid(_grid)
        plt.show(block=False)
        return magnitude_spectrum
    else: 
        raise ValueError('spectrum is None')


def estimate_psd(spec):
    return (np.abs(spec) ** 2).astype(np.float32)


def plot_couple_img_spect(img, spectrum=None, titles=('', ''), _grid=False):
    if img is not None:
        if spectrum is not None:
            magnitude_spectrum = 20*np.log(np.abs(spectrum))
        else:
            magnitude_spectrum = 20*np.log(np.abs(fft(img)))
    
    plt.figure(figsize=(14,14))
    plt.subplot(121)
    plt.grid(_grid)
    plt.imshow(img, cmap='gray')
    plt.title(titles[0])
    plt.subplot(122)
    plt.grid(_grid)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(titles[1])
    plt.show(block=False)
    return magnitude_spectrum


def fft_3d_to_cube(parall, num_of_slices_P, block_side):
    # 3D FFT

    # Spectrum creation:
    # due to the resolution_factor rf = (res(Z) / res(X,Y))
    # => parallelepiped has shape (rf*n, rf*n, n), n = num_of_slices_P.
    # For example: n = 8, rf = 4.55
    # => Spectrum has same shape (36, 36, 8).
    # For frequency filtering, we create fake cube spectrum with
    # [  (36 - 8)/2  ] slices of (36 * 36) zeros => slices_pad = 3n / 2
    # [       8      ] slices of real data
    # [  (36 - 8)/2  ] slices of (36 * 36) zeros
    # with shape (36, 36, 36) = (rf*n, rf*n, rf*n)
    #
    slices_pad = int((block_side - num_of_slices_P) / 2)

    # dimension of cube
    cube_shape = [block_side] * 3

    # preallocate empty psd and empty cubes for spectrum_cube and psd_cube
    psd = np.empty_like(parall)
    psd_cube = np.zeros(cube_shape, dtype=np.float32)
    spec_cube = np.zeros(cube_shape, dtype=np.complex128)

    # estimate spectrum and save it inside cube
    spectrum = np.fft.fftshift(np.fft.fftn(parall))
    spec_cube[:, :, slices_pad: slices_pad + num_of_slices_P] = spectrum

    # etimate psd of parallelogram and saved inside psd_cube
    noise = np.finfo(spectrum.dtype).eps  # adds eps to spectrum for to fix logarithm dominium limit
    psd = estimate_psd(spectrum + noise)
    psd_cube[:, :, slices_pad: slices_pad + num_of_slices_P] = np.copy(psd)

    return spec_cube, psd, psd_cube, slices_pad


def find_centroid_of_peak(psd, peak_coords):
    n = 2  # side/2 of window: [x-n:x+n+1, y-n:y+n+1, z-n:z+n+1] -> [2n+1, 2n+1, 2n+1]

    (x_max, y_max, z_max) = peak_coords

    # define 3d window centered in peak_coords  on which calculate centroid
    window = psd[
             x_max - n: x_max + n + 1,
             y_max - n: y_max + n + 1,
             z_max - n: z_max + n + 1]

    # scipy.ndimage.measurements.center_of_mass
    # coordinate of center of mass in window coordinates system
    (x_cm, y_cm, z_cm) = center_of_mass(window)

    # change coordinate from window system to psd system
    centroid_coords = (
        x_cm + x_max - n,
        y_cm + y_max - n,
        z_cm + z_max - n)
    return centroid_coords


def find_peak_in_psd(psd):
    # find max value in psd matrix, and return:
    # 100 * max_value / psd.sum
    # tuple (r,c,z) of unravel coordinates of max finded, in psd matrix system
    #
    max_coords = np.unravel_index(psd.argmax(), psd.shape)
    psd_sum = np.sum(psd)
    
    # 1 - valore del picco / integrale
    peak_value = psd[max_coords]
    peak_ratio = 100 * peak_value / psd_sum
    
    # deprecated
    # # 2 - (somma dei valori dell'intorno 3x3x3 del picco) / integrale
    # n = 1  # side/2 of window: [x-n:x+n+1, y-n:y+n+1, z-n:z+n+1] -> [2n+1, 2n+1, 2n+1]
    # (x_max, y_max, z_max) = max_coords
    # # define 3d window centered in max_coords on which calculate sum
    # neighbours = psd[
    #          x_max - n: x_max + n + 1,
    #          y_max - n: y_max + n + 1,
    #          z_max - n: z_max + n + 1]
    # neighbours_sum = np.sum(neighbours)
    #
    # # 3 - normalizzazione rispetto alla somma del contenuto spettrale in quelle frequenze
    # peak_ratio_neigh = 100 * neighbours_sum / psd_sum
    # return max_coords, peak_ratio, peak_ratio_neigh

    return max_coords, peak_ratio


def create_3D_filter(block_side, res_xy, sarc_length):
    # CREATE SPHERICAL MASK

    # res_xy: x,y resolution
    # block_side: side of every slice in pixel
    # Frequency of sarceomeres in Pixel:
    sarc_pixel_period = (block_side * res_xy) / sarc_length  # [pixel * um / um] = [pixel]

    # Band-pass filter for that structures == spherical shell with:
    radius = sarc_pixel_period  # fc
    band_width = int(block_side / 16)  # thickness of shell in pixels (if size = 32 pix -> band_widt = 2 pix)
    f2 = np.ceil(radius + (band_width / 2))
    f1 = np.ceil(radius - (band_width / 2))

    # position = 0.5 for centering the sphere
    sphere_f2 = (geom.sphere(shape=block_side, radius=f2, position=0.5)).astype(bool)
    sphere_f1 = ~(geom.sphere(shape=block_side, radius=f1, position=0.5)).astype(bool)
    mask = (sphere_f2 * sphere_f1).astype(np.bool)
    return mask


def filtering_3D(spectrum_cube, mask):
    spec_cube_filt = spectrum_cube * mask
    psd_cube_filt = estimate_psd(spec_cube_filt)

    return spec_cube_filt, psd_cube_filt


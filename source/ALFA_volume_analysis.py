import argparse
import cv2
import numpy as np
import os
import time

from skimage import exposure

# local modules
from custom_tool_kit import search_value_in_txt, seconds_to_hour_min_sec
from make_data import manage_path_argument, load_tif_data
from custom_image_tool import normalize, save_tiff, image_have_info, create_img_name_from_index
from custom_thresholds import opencv_th_k_means_th, make_cells_mask, widens_mask_deconv


def main(parser):

    # --- PRELIMINARY OPERATIONS --------------------------------------------------------

    # read args from console
    args = parser.parse_args()
    source_path = manage_path_argument(args.source_folder)

    # take base path and stack name
    base_path = os.path.dirname(os.path.dirname(source_path))
    stack_name = os.path.basename(source_path)

    # Def .txt filepath
    txt_parameters_path = os.path.join(base_path, 'parameters.txt')
    txt_results_path = os.path.join(base_path, 'ALFA_volume_results.txt')
    txt_metadata_path = os.path.join(base_path, 'metadata.txt')

    process_names = ['clahe', 'mask_bin', 'segmented', 'contourned', 'contours']

    # create destination paths where save images
    destination_paths = []
    for process in process_names:
        destination_paths.append(os.path.join(base_path, process, stack_name))

    # if those folders do not exist, this creates them
    for path in destination_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    # --- SCRIPT ---------------------------------------------------------------

    # reads parameters
    parameters = extract_parameters(txt_parameters_path)

    # read user preference (if save images or not)
    _save_clahe = bool(parameters['save_clahe'])
    _save_binary_mask = bool(parameters['save_binary_mask'])
    _save_segmented = bool(parameters['save_segmented'])
    _save_countourned = bool(parameters['save_countourned'])
    _save_contours = bool(parameters['save_contours'])

    # print to video and write in results.txt init message
    init_message = [
                    ' Source from path : {} \n'.format(base_path),
                    ' Stack : {} \n'.format(stack_name),
                    '\n\n - Start load the Stack, this may take a few minutes... '
                    ]
    error_message = '\n *** ERROR *** : stack in this path is None'

    with open(txt_results_path, 'w') as f:
        for line in init_message:
            print(line, end='')
            f.write(line+'\n')

    # measure units
    x_step = parameters['res_xy'] # micron
    y_step = parameters['res_xy']  # micron
    z_step = parameters['res_z']  # micron
    voxel_in_micron3 = x_step * y_step * z_step  # micron^3

    # extract stack (OLD METHOD)
    # volume, mess = load_stack_into_numpy_ndarray([source_path])
    # img_shape = volume[:, :, 0].shape

    # extract stack (NEW METHOD)---------------
    volume = load_tif_data(source_path)
    if len(volume.shape) == 2:
        volume = np.expand_dims(volume, axis=2)  # add the zeta axis
    img_shape = (volume.shape[0], volume.shape[1])
    # ---------------------------------------------

    print(' Images shape : ', img_shape, '\n')

    # measure of imaging volume
    area_of_slice = img_shape[0] * img_shape[1]
    number_of_slices = volume.shape[2]

    # Estimated Volume
    total_imaging_volume = area_of_slice * number_of_slices  # total volume, from z=0 to z maximum

    # Initializing to zero the Volume counter
    effective_myocites_volume = 0  # real estimated volume of myocites (sum of area of real cells)
    
    # Boolean vector with length = number_of_slices. 
    # Element i-th is:
    # True - if i-th slice have info
    # False - if i-th lice i empty
    slices_info = np.zeros(number_of_slices).astype(bool)  # created with all False values.

    # save elaboration time for each iteration (for estimate mean time of elaboration)
    slice_elab_time_list = list()
    t_start = time.time()

    print(' EQUALIZATION and SEGMENTATION of every frame:')

    with open(txt_results_path, 'a') as f:
        if volume is not None:

            for z in range(number_of_slices):

                # extract current slice
                img = volume[:, :, z]
                img_name = create_img_name_from_index(z)  # img_name.tif

                #  check if img is empty (too black)
                if image_have_info(img, parameters['t_rate_info']):
                    elab_start = time.time()

                    slices_info[z] = True
                    img = normalize(img)

                    equalized_img = clahe(img, parameters, img_name, destination_paths[0], _save=_save_clahe)

                    bw_mask, pixels_of_real_cells = create_byn_mask(equalized_img, parameters, img_name, destination_paths[1], _save=_save_binary_mask)

                    effective_myocites_volume += pixels_of_real_cells

                    create_and_save_segmented_images(bw_mask, equalized_img, img_name, destination_paths[2], _save=_save_segmented)

                    contours = create_surrounded_images(bw_mask, equalized_img, img_name, destination_paths[3], _save=_save_countourned)

                    if _save_contours:
                        # save image with only contours for fiji visualization:
                        save_contours(img_name, contours, bw_mask.shape, destination_paths[4])
                        save_contours(img_name, contours, bw_mask.shape, destination_paths[4])

                    slice_elab_time_list.append(time.time() - elab_start)            
                    elapsed_time = (number_of_slices - z - 1) * np.mean(slice_elab_time_list)
                    (h, m, s) = seconds_to_hour_min_sec(elapsed_time)
                    
                    measure = ' - {0} --> {1:.1f} um^3   -   ET: {2:2d}h {3:2d}m {4:2d}s'.format(img_name,
                                                                                                 pixels_of_real_cells * voxel_in_micron3,
                                                                                                 int(h), int(m), int(s))

                else:
                    # black = np.zeros(img.shape)
                    # for path in destination_paths:
                    #     save_tiff(img=black, img_name=img_name, comment='empty', folder_path=path)
                    measure = ' - {} is black, rejected'.format(img_name)

                print(measure)
                f.write(measure + '\n')

            # execution time
            (h, m, s) = seconds_to_hour_min_sec(time.time() - t_start)

            # Num of saved slices
            saved = np.count_nonzero(slices_info)

            # Num of empty slice on top = Index of first slice with info
            empty_on_top = np.where(slices_info==True)[0][0]

            # Num of empty slice on bottom = Index of first slice with info, searching from from z=z_max to z=0
            empty_on_bottom = np.where(np.flipud(slices_info)==True)[0][0]

            # volumes in micron^3
            total_imaging_volume_in_micron = total_imaging_volume * voxel_in_micron3
            effective_volume_in_micron3 = effective_myocites_volume * voxel_in_micron3

            # in percentage
            myocites_perc = 100 * effective_myocites_volume / total_imaging_volume

            result_message = list()
            result_message.append('\n ***  Process successfully completed, time of execution: {0:2d}h {1:2d}m {2:2d}s \n'.format(int(h), int(m), int(s)))
            result_message.append(' Number of saved slices: {}'.format(saved))
            result_message.append(' Number of rejected slices (because empty) on the top of Volume: {}'.format(empty_on_top))
            result_message.append(' Number of rejected slices (because empty) on the bottom of Volume: {}'.format(empty_on_bottom))
            result_message.append(' Total Imaging volume : {0:.6f} mm^3'.format(total_imaging_volume_in_micron / 10**9))
            result_message.append(' Effective miocytes tissue volume : {0:.6f} mm^3, {1:.3f}% of Imaging volume'.format(effective_volume_in_micron3 / 10**9, myocites_perc))
            result_message.append('\n')
            result_message.append(' OUTPUT SAVED IN: \n')

            for path in destination_paths:
                result_message.append(path)

            # write and print results
            for l in result_message:
                print(l)
                f.write(l+'\n')

            # write metadata
            with open(txt_metadata_path, 'w') as m:
                m.write('empty_on_top = {}\n'.format(empty_on_top))
                m.write('empty_on_bottom = {}\n'.format(empty_on_bottom))
                m.write('saved = {}'.format(saved))

        else:
            print(error_message)
            f.write(error_message)
            
        print(' \n \n \n ')

# ================================ END MAIN ===================================================


def extract_parameters(filename):
    ''' read parameters values in filename.txt
    and save it in a dictionary'''

    param_names = ['res_xy',
                   'res_z',
                   't_rate_info',
                   'clahe_ksize',
                   'clip_clahe',
                   'K_cluster',
                   'K_true',
                   't_ratio_holes',
                   'save_clahe',
                   'save_binary_mask',
                   'save_segmented',
                   'save_contours',
                   'save_countourned']

    # read values in txt
    param_values = search_value_in_txt(filename, param_names)

    print(' ***  Parameters : \n')
    # create dictionary of parameters
    parameters = {}
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        print(' - {} : {}'.format(p_name, param_values[i]))
    print('\n \n')
    return parameters


def save_contours(img_name, contours, img_shape, countours_folder):
    # empty black image
    black = np.zeros(img_shape)

    # draw contours
    cv2.drawContours(black, contours, -1, 255, 1)

    # save singular images
    save_tiff(img=black, img_name=img_name, comment='only_cont', folder_path=countours_folder)


def clahe(img, parameters, img_name, to_folder, _save=False):
    ksize = parameters['clahe_ksize']
    clip = parameters['clip_clahe']
    img = normalize(exposure.equalize_adapthist(img, clip_limit=clip, kernel_size=ksize))
    if _save:
        save_tiff(img=img, img_name=img_name, comment='eq', folder_path=to_folder)
    return img


def create_byn_mask(img_eq, parameters, img_name, to_byn_folder, _save=False):
    # opencv k-means clustering segmentation
    K = int(parameters['K_cluster'])
    img_k4, center, label = opencv_th_k_means_th(img=img_eq, K=K)

    # choice label for segmentation
    k_true_by_param = int(parameters['K_true'])
    unique, counts = np.unique(img_k4, return_counts=True)

    # check cluster dimension
    if counts[1] > counts[0]:
        k_true = k_true_by_param - 1  # 2th cluster is too big -> 1th and 2th are mapped to black
    else:
        k_true = k_true_by_param  # 2th cluster is ok -> only 1th cluster is mapped to black

    # create binary mask from segmented images
    cells_mask = make_cells_mask(center, label, img_eq.shape, K=K, true=k_true)  # pixels are ZERO or ONE

    # prepare defintive mask
    t_ratio_holes = parameters['t_ratio_holes']
    BW_cells_mask = widens_mask_deconv(cells_mask, t_ratio_holes)

    if _save:
        save_tiff(img=BW_cells_mask, img_name=img_name, comment='bin', folder_path=to_byn_folder)
    return BW_cells_mask, np.count_nonzero(BW_cells_mask)


def create_and_save_segmented_images(bw_cells_mask, img_eq, img_name, to_seg_folder, _save=False):
    final = img_eq * bw_cells_mask.astype(bool)
    if _save:
        save_tiff(img=final, img_name=img_name, comment='seg', folder_path=to_seg_folder)


def create_surrounded_images(BW_mask, img_eq, img_name, to_countourned_folder,  _save=False):
    # plot contour mak over original image
    contours = (cv2.findContours(BW_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))[1]

    # prepare image
    img_eq_with_contour = normalize(img_eq.copy())

    # draw contour
    cv2.drawContours(img_eq_with_contour, contours, -1, 255, 1)

    if _save:
        save_tiff(img=img_eq_with_contour, img_name=img_name, comment='cont', folder_path=to_countourned_folder)
    return contours


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation of images in source folder')
    parser.add_argument('-sf', '--source-folder', nargs='+', help='Images to segment', required=False)

    main(parser)

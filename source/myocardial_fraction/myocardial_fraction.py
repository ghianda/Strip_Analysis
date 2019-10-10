import argparse
import numpy as np
import os
import time

from scipy import ndimage
from skimage.morphology import convex_hull_image

# local modules
from custom_tool_kit import search_value_in_txt, seconds_to_min_sec
from custom_tool_kit import manage_path_argument
from custom_image_base_tool import save_tiff, create_img_name_from_index
from zetastitcher import InputFile


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main(parser):

    # read args from console
    args = parser.parse_args()

    # read source path (path of binary segmentation images)
    source_path = manage_path_argument(args.source_folder)

    # take base path and stack name
    base_path = os.path.dirname(os.path.dirname(source_path))
    stack_name = os.path.basename(source_path)

    # create path and folder where save section images
    sections_path = os.path.join(base_path, 'xz_sections', stack_name)
    filled_sections_path = os.path.join(base_path, 'xz_filled_sections', stack_name)
    convex_sections_path = os.path.join(base_path, 'xz_convex_sections', stack_name)
    for path in [sections_path, filled_sections_path, convex_sections_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # portion of y axes for section estimation
    y_start = np.uint16(args.y_start[0])
    y_stop = np.uint16(args.y_stop[0])

    # Def .txt filepath
    txt_parameters_path = os.path.join(base_path, 'parameters.txt')
    txt_results_path = os.path.join(base_path, 'Measure_analysis.txt')

    # SCRIPT ----------------------------------------------------------------------
    
    # print to video and write in results.txt init message
    init_message = [' ****  Script for Estimation of Real Myocardial fraction volume **** \n \n'
                    ' Source from path : {}'.format(base_path),
                    ' Stack : {}'.format(stack_name),
                    '\n\n *** Start processing... \n'
                    ]
    error_message = '\n *** ERROR *** : stack in this path is None'
    with open(txt_results_path, 'w') as f:
        for line in init_message:
            print(line)
            f.write(line+'\n')

    # reads parameters
    parameters = extract_parameters(txt_parameters_path)

    # measure units
    x_step = parameters['res_xy']  # micron
    y_step = parameters['res_xy']  # micron
    z_step = parameters['res_z']  # micron
    pixel_xz_in_micron2 = x_step * z_step  # micron^2
    voxel_in_micron3 = x_step * y_step * z_step  # micron^3

    # preferences
    _save_binary_sections = bool(parameters['save_binary_sections'])

    # load data
    print(' *** Start to load the Stack...')
    infile = InputFile(source_path)
    masks = infile.whole()

    # swap axis from ZYX to YXZ
    masks = np.moveaxis(masks, 0, -1)

    # check if it's a 3D or a 2D image (if only one frame, it's 2D and i add an empty axis
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=2)  # add the zeta axis

    # count selected sections
    total_sections = masks.shape[0]  # row -> y -> number of sections
    print('\n Volume shape:', masks.shape)
    print('Number of total sections: ', total_sections)
    print('\n')

    # set y portion [optional]
    if y_start == y_start == 0:
        y_start = 0
        y_stop = total_sections - 1
        print(' *** ATTENTION : selected all the sections: {} -> {}'.format(y_start, y_stop))
    if y_stop < y_start:
        y_start = np.uint16(total_sections / 4)
        y_stop = np.uint16(total_sections * 3 / 4)
        print(' *** ATTENTION : y portion selected by DEFAULT: {} -> {}'.format(y_start, y_stop))

    # Every Section(y) is XZ projection of mask. Estimated Area is the sum of the Non_Zero pixel in the section image
    selected_sections = np.uint16(y_stop - y_start)
    sections_micron2 = np.zeros(selected_sections) # area in micron of every section
    print('Number of selected sections: ', y_stop-y_start)

    # Initializing to zero the Volume counters
    effective_myocites_volume = 0  # real estimated volume of myocites (sum of area of real cells in sections)
    filled_myocite_volume = 0  # filled tissue volume (sum of area of the sections with filled holes)
    global_tissue_volume = 0  # global tissue volume (sum of area of convex envelop of the sections)

    t_start = time.time()
    analyzed_section = 0  # counter, only for control the for loop (remove)

    with open(txt_results_path, 'a') as f:
        
        pre_info = list()
        pre_info.append('\nPortion of y selected: [{} -> {}]'.format(y_start, y_stop))
        pre_info.append('Option for save the sections images: {}'.format(_save_binary_sections))
        pre_info.append('\n')
        for l in pre_info:
            print(l)
            f.write(l+'\n')

        if masks is not None:

            print('\n... Estimation of mean section and Volume fraction of Myocardial Tissue...')
            for y in range(y_start, y_stop):

                # extract section
                section = masks[y, :, :]
                sec_name = create_img_name_from_index(total_sections - y - 1)  # img_name.tif

                # count pixels of real cardiomyocyte cells of current section
                pixels_with_cardiomyocyte = np.count_nonzero(section)
                effective_myocites_volume += pixels_with_cardiomyocyte

                # save original sections
                if _save_binary_sections:
                    # transform point of view and save
                    save_tiff(img=np.rot90(m=np.flipud(section), k=1, axes=(0, 1)),
                              img_name=sec_name, comment='section', folder_path=sections_path)

                # fill the section holes and set comment for tiff filenames (to save images)
                section = 255 * ndimage.morphology.binary_fill_holes(section).astype(np.uint8)
                # count cell pixel in the envelopped section
                pixels_with_filled_cell = np.count_nonzero(section.astype(bool))
                filled_myocite_volume += pixels_with_filled_cell

                if _save_binary_sections:
                    # transform point of view and save
                    save_tiff(img=np.rot90(m=np.flipud(section), k=1, axes=(0, 1)),
                              img_name=sec_name, comment='filled_section', folder_path=filled_sections_path)

                # create envelop (convex polygon) of section to estimate and set comment for tiff filenames
                section = 255 * convex_hull_image(np.ascontiguousarray(section)).astype(np.uint8)  # envelop
                if _save_binary_sections:
                    # transform point of view and save
                    save_tiff(img=np.rot90(m=np.flipud(section), k=1, axes=(0, 1)),
                              img_name=sec_name, comment='convex_section', folder_path=convex_sections_path)

                # count cell pixel in the enveloped section
                pixels_with_generic_cell = np.count_nonzero(section.astype(bool))
                global_tissue_volume += pixels_with_generic_cell

                # estimate area of this section
                if pixels_with_cardiomyocyte > 0:
                    real_area_in_micron2 = pixels_with_cardiomyocyte * pixel_xz_in_micron2
                    filled_area_in_micron2 = pixels_with_filled_cell * pixel_xz_in_micron2
                    global_area_in_micron2 = pixels_with_generic_cell * pixel_xz_in_micron2

                    # save in the section area list
                    sections_micron2[y - y_start] = real_area_in_micron2

                    # create string messages
                    measure = bcolors.OKBLUE + '{}'.format(os.path.basename(base_path)) + bcolors.ENDC + \
                              ' - {} ->'.format(sec_name) + \
                              'real: {0:3.1f} um^2 - filled: {1:3.1f} um^2 - convex: {2:3.1f}'.\
                                  format(real_area_in_micron2, filled_area_in_micron2, global_area_in_micron2)

                else:
                    measure = ' - {} is empty'.format(sec_name)

                analyzed_section += 1
                print(measure)
                # f.write(measure+'\n')

            # execution time
            (h, m, s) = seconds_to_min_sec(time.time() - t_start)

            # percentage of cardiomyocyte volumes
            perc_fill = 100 * effective_myocites_volume / filled_myocite_volume
            perc_env = 100 * effective_myocites_volume / global_tissue_volume

            # volumes in micron^3
            effective_volume_in_micron3 = effective_myocites_volume * voxel_in_micron3
            filled_volume_in_micron3 = filled_myocite_volume * voxel_in_micron3
            global_tissue_volume_in_micron3 = global_tissue_volume * voxel_in_micron3

            # count empty sections
            sections_with_cell = np.count_nonzero(sections_micron2)
            empties = selected_sections - sections_with_cell

            # Mean sections:   
            mean_section = np.sum(sections_micron2) / sections_with_cell  # (original images)

            # create results string
            result_message = list()
            result_message.append('\n ***  Process successfully completed, time of execution: {0:2d}h {1:2d}m {2:2d}s \n'.format(int(h), int(m), int(s)))
            result_message.append(' Total number of frames: {}'.format(masks.shape[2]))
            result_message.append(' Total sections: {}'.format(total_sections))
            result_message.append(' Selected sections: {}'.format(selected_sections))
            result_message.append(' Effective analyzed sections: {}'.format(analyzed_section))
            result_message.append(' Number of empty section: {}'.format(empties))
            result_message.append(' Number of section with cells: {}'.format(sections_with_cell))
            result_message.append('\n')
            result_message.append(' Mean sections: {0:.3f} um^2'.format(mean_section))
            result_message.append('\n')
            result_message.append(' Myocardium volume : {0:.6f} mm^3'.format(effective_volume_in_micron3 / 10 ** 9))
            result_message.append(' Filled volume : {0:.6f} mm^3'.format(filled_volume_in_micron3 / 10 ** 9))
            result_message.append(' Global volume : {0:.6f} mm^3'.format(global_tissue_volume_in_micron3 / 10 ** 9))
            result_message.append(' Percentage of myocardium tissue filled: {}%'.format(perc_fill))
            result_message.append(' Percentage of myocardium tissue enveloped: {}%'.format(perc_env))

            result_message.append('\n')
            result_message.append(' \n OUTPUT SAVED IN: \n')
            result_message.append(txt_results_path)

            # write and print results
            for l in result_message:
                print(l)
                f.write(l+'\n')

        else:
            print(error_message)
            f.write(error_message)

        print(' \n \n \n ')
    
# ==================================== END MAIN =========================================== """


def extract_parameters(filename):
    ''' read parameters values in filename.txt
    and save it in a dictionary'''

    param_names = ['res_xy',
                   'res_z',
                   'save_binary_sections']

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for Estimation of Mean Section')
    parser.add_argument('-sf', '--source-folder', nargs='+', help='Slices or Stack to load', required=False)
    parser.add_argument('-y1', '--y-start', nargs='+', help='y start for section estimation', required=False)
    parser.add_argument('-y2', '--y-stop', nargs='+', help='y stop for section estimation', required=False)

    main(parser)
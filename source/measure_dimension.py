''' Difference with original BETA function:
1) take parameters y1 and y2 (portion of y for estimate the section mean)
2) evaluate both original and filled section area: for the filled, fill the holes for every section and count pixel
3) evaluate complete volume

1) assicurarsi che lo stack mask_bin contenga le immagini segmentate
2) se si vuole salvare le sezioni segmentate, assicurarsi che in parameters.txt l'opzione 'save_binary_section' sia 1
3) aprise lo stack 'segmented' con imagej (se serve, creare la proiezione in z con average intensity per vedere megio struttura strip) e decidere la porzione dell'asse y da analizzare (y_start e y_stop) e decidere se serve riempire i bchi (se manca marcatura) oppure no
4) eseguire measure_dimension.py così:
python3 measure_dimension.py -sf source_folder -y1 y_start -y2 y_stop -fill x
con
 - source_folder : pat dello stack segmentato binario ('.../.../nome_campione/mask_bin/stitched_stack/')
 - y_start e y_stop : porzione centrale strip in cui stimare la sezione media
  - x : 0 se non si vogliono riempire buchi, 1 se si vogliono riempire

  lo script produce il file 'Measure_analysis.txt' in cui riporta la sezione media e il volume stimati (oltre al report delle opzioni scelte)
'''



import argparse
import numpy as np
import os
import time

from scipy import ndimage

# local modules
from custom_tool_kit import search_value_in_txt, seconds_to_min_sec
from custom_image_tool import save_tiff, create_img_name_from_index

# old loader:
from make_data import load_stack_into_numpy_ndarray, manage_path_argument, load_tif_data

# new loader:
from make_data import load_tif_data, manage_path_argument


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
    for path in [sections_path, filled_sections_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # portion of y axes for section estimation
    y_start = np.uint16(args.y_start[0])
    y_stop = np.uint16(args.y_stop[0])

    # preference for fill or not the holes
    _fill_holes = args.fill_holes

    print('\n\n\n _fill_holes : ', _fill_holes)

    # Def .txt filepath
    txt_parameters_path = os.path.join(base_path, 'parameters.txt')
    if _fill_holes:
        txt_results_path = os.path.join(base_path, 'Measure_analysis_filled.txt')
    else:
        txt_results_path = os.path.join(base_path, 'Measure_analysis.txt')

    # SCRIPT ----------------------------------------------------------------------
    
    # print to video and write in results.txt init message
    init_message = [' ****  Script for Estimation of Section Area and Volume **** \n \n'
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
    pixel_xy_in_micron2 = x_step * y_step  # micron^2
    pixel_xz_in_micron2 = x_step * z_step  # micron^2
    voxel_in_micron3 = x_step * y_step * z_step  # micron^3

    # preferences
    _save_binary_sections = bool(parameters['save_binary_sections'])

    # create images stack from source path
    print(' *** Start to load the Stack...')

    # old version:
    # masks, mess = load_stack_into_numpy_ndarray([source_path])
    # print(mess)

    #new version:
    masks = load_tif_data(source_path)
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=2)  # add the zeta axis

    # count selected sections
    total_sections = masks.shape[0]  # row -> y -> number of sections
    print('\n Volume shape:', masks.shape)
    print('Number of total sections: ', total_sections)
    print('\n')

    # set y portion
    if y_stop < y_start:
        y_start = np.uint16(total_sections / 4)
        y_stop = np.uint16(total_sections * 3 / 4)
        print(' *** ATTENTION : y portion selected by DEFAULT: {} -> {}'.format(y_start, y_stop))

    # Every Section(y) is XZ projection of mask. Estimated Area is the sum of the Non_Zero pixel
    selected_sections = np.uint16(y_stop - y_start)
    sections_micron2 = np.zeros(selected_sections)
    print('Number of selected sections: ', y_stop-y_start)

    # Initializing to zero the Volume counter
    effective_myocites_volume = 0  # real estimated volume of myocites (sum of area of real cells)

    t_start = time.time()
    analyzed_section = 0  # counter, only for control the for loop (remove)

    with open(txt_results_path, 'a') as f:
        
        pre_info = list()
        pre_info.append('\nPortion of y selected: [{} -> {}]'.format(y_start, y_stop))
        pre_info.append('Option for fill the holes: {}'.format(_fill_holes))
        pre_info.append('\n')
        for l in pre_info:
            print(l)
            f.write(l+'\n')

        if masks is not None:
            
            # *** 1 ***  - Mean Section Estimation
            print('\n... Estimation of mean section...')
            for y in range(y_start, y_stop):

                # extract section
                section = masks[y, :, :]
                sec_name = create_img_name_from_index(total_sections - y - 1)  # img_name.tif

                # filled the section holes and set comment for tiff filenames
                if _fill_holes:
                    section = ndimage.morphology.binary_fill_holes(section)
                    comment = 'filled_section'
                    dest_path = filled_sections_path
                else: 
                    comment = 'section'
                    dest_path = sections_path

                # count cell pixel
                pixels_with_cell = np.count_nonzero(section.astype(bool))

                if pixels_with_cell > 0:
                    area_in_micron2 = pixels_with_cell * pixel_xz_in_micron2
                    sections_micron2[y - y_start] = area_in_micron2
                    measure = ' - {}   ---->   {} um^2'.format(sec_name, area_in_micron2)

                    if _save_binary_sections:
                        # transform point of view
                        section = np.rot90(m=np.flipud(section), k=1, axes=(0, 1))
                        # save section in correct path
                        save_tiff(img=section, img_name=sec_name, comment=comment, folder_path=dest_path)
                
                else:
                    measure = ' - {} is empty'.format(sec_name)

                analyzed_section += 1
                print(measure)
                # f.write(measure+'\n')

            # *** 2 ***  - Volume Estimation
            print('\n\n ...Estimation of Volume...\n')
            for z in range(masks.shape[2]):
                # check fill option
                if _fill_holes:
                    print( '... filling {} xy frame'.format(z))
                    z_frame = ndimage.morphology.binary_fill_holes(masks[:, :, z])
                else:
                    z_frame = masks[:, :, z]
                # add pixels_of_real_cells of current z-slice to the counter
                effective_myocites_volume += np.count_nonzero(z_frame)

            # execution time
            (h, m, s) = seconds_to_min_sec(time.time() - t_start)

            # volumes in micron^3
            effective_volume_in_micron3 = effective_myocites_volume * voxel_in_micron3

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
            result_message.append(' Effective miocytes tissue volume : {0:.6f} mm^3'.format(effective_volume_in_micron3 / 10**9))

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
    parser.add_argument('--fill', action='store_true', dest='fill_holes', help='Fill the holes inside images')

    main(parser)
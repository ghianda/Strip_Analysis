import argparse
import numpy as np
import os
import time

# local modules
from custom_tool_kit import search_value_in_txt, seconds_to_min_sec
from make_data import load_stack_into_numpy_ndarray, manage_path_argument
from custom_image_tool import save_tiff, create_img_name_from_index


def main(parser):

    # read args from console
    args = parser.parse_args()

    source_path = manage_path_argument(args.source_folder)

    # source_path = args.source_folder

    # if type(source_path) is list:
    #     if len(source_path) > 1:
    #         given_path = ' '.join(source_path)
    #         source_path = [given_path]  # prepare source_path for 'make_dataset' function (it takes a list in input)
    #     else:
    #         given_path = source_path[0]
    # else:
    #     given_path = source_path

    # # remove last backlslash
    # if given_path.endswith('/'):
    #     given_path = given_path[0:-1]

    # take base path and stack name
    base_path = os.path.dirname(os.path.dirname(source_path))
    stack_name = os.path.basename(source_path)

    # Def .txt filepath
    txt_parameters_path = os.path.join(base_path, 'parameters.txt')
    txt_results_path = os.path.join(base_path, 'BETA_section_results.txt')

    # create paths for load masked images
    # mask_path = os.path.join(base_path, 'mask_bin', stack_name)

    # create path and folder where save section images
    sections_path = os.path.join(base_path, 'xz_sections', stack_name)
    if not os.path.exists(sections_path):
        os.makedirs(sections_path)

    # SCRIPT -----------------------------------------

    # print to video and write in results.txt init message
    init_message = [' ****  Script for Estimation of Section Area **** \n \n'
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

    # create images stack from source path
    print(' *** Start to load the Stack...')
    masks, mess = load_stack_into_numpy_ndarray([source_path])
    print(mess)
    # source_data = make_dataset(source_path)
    # data_length = len(source_data)

    # masks = create_stack_light(source_data)

    # if data_length == masks.shape[2]:
    #     print(' Stack loaded successfully')
    # else:
    #     print(' *** WARNING -> len(image_list) != slices in ndarray -> check loading')

    # del source_data

    number_of_sections = masks.shape[0]  # row -> y -> number of sections
    print('number_of_sections', number_of_sections)
    print('\n shape:', masks.shape)

    # list of sections Area.
    # Every Section(y) is XZ projection of mask, the sum of the Non_Zero pixel is the estimated Area
    sections_micron2 = np.zeros(number_of_sections)

    t_start = time.time()

    with open(txt_results_path, 'a') as f:
        if masks is not None:
            for y in range(number_of_sections):

                section = masks[y, :, :]
                sec_name = create_img_name_from_index(number_of_sections - y - 1)  # img_name.tif

                # count cell pixel
                pixels_with_cell = np.count_nonzero(section.astype(bool))
                
                if pixels_with_cell > 0 :
                    area_in_micron2 = pixels_with_cell * pixel_xz_in_micron2
                    sections_micron2[y] = area_in_micron2
                    measure = ' - {} --> {} um^2'.format(sec_name, area_in_micron2)

                    # transform point of view
                    section = np.rot90(m=section, k=1, axes=(0,1))
                    section = np.flipud(section)

                    # save image of section
                    save_tiff(img=section, img_name=sec_name, comment='section', folder_path=sections_path)
                
                else:
                    measure = ' - {} is empty'.format(sec_name)

                print(measure)
                # f.write(measure+'\n')

            # execution time
            (h, m, s) = seconds_to_min_sec(time.time() - t_start)

            # count empty sections
            sections_with_cell = np.count_nonzero(sections_micron2)
            empties = number_of_sections - sections_with_cell

            # Mean sections
            mean_section = np.sum(sections_micron2) / sections_with_cell

            # create results string
            result_message = list()
            result_message.append('\n ***  Process successfully completed, time of execution: {0:2d}h {1:2d}m {2:2d}s \n'.format(int(h), int(m), int(s)))
            result_message.append(' Number of empty section: {}'.format(empties))
            result_message.append(' Number of section with cells: {}'.format(sections_with_cell))
            result_message.append('\n')
            result_message.append(' Mean sections : {0:.3f} um^2'.format(mean_section))
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
                   'res_z']

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
    main(parser)
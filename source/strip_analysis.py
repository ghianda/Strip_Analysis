import argparse
import os
from make_data import manage_path_argument

''' 
Main Script of Structural Analysis

parameters: source_folder (of deconvolted images)


Execute:

bla bla bla

'''


def structural_analysis(parser):

    print('\n\n')
    print(' *********************************************************************************')
    print(' ************************* Structural_analysis.py script *************************')
    print(' *********************************************************************************')
    print('\n')

    args = parser.parse_args()

    # extract path string
    source_path = manage_path_argument(args.source_folder)
    source_path = source_path.replace(' ', '\ ')  # correct whitespace with backslash

    # extract base path and stack name
    base_path = os.path.dirname(os.path.dirname(source_path))
    stack_name = os.path.basename(source_path)

    # create folder path of binary mask images
    mask_path = os.path.join(base_path, 'mask_bin', stack_name)

    # create folder path of segmented images
    segmented_path = os.path.join(base_path, 'segmented', stack_name)

    # call  ALFA_volume_analysis script
    #print(' ---------> call ALFA')
    os.system('python3 ALFA_volume_analysis.py -sf {}'.format(source_path))

    # call  BETA_estimated_section script
    #print(' ---------> call BETA')
    os.system('python3 BETA_estimated_section.py -sf {}'.format(mask_path))

    # call  ALFA_volume_analysis script
    #print(' ---------> call GAMMA')
    os.system('python3 GAMMA_orientation_analysis.py -sf {}'.format(segmented_path))


def main():

    # define parser
    parser = argparse.ArgumentParser(
        description='Strip images enhanced and 3D Structural analysis',
        epilog='Author: Francesco Giardini <giardini@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # command line REQUIRED argument
    parser.add_argument('-sf', '--source_folder', nargs='+', required=False, help='input images path')

    # command line OPTIONAL argument, passed without value (if passed '--ss', args.save_sections is True, else False)
    parser.add_argument('--ss', action='store_true', dest='save_sections', help='save binary sections images on disk')
    # NOTE - THIS IS A TEST -> SCRIPT NOT USE -ss OPTION, BUT READ parameters.txt FILE

    # run analyis
    structural_analysis(parser)

if __name__ == '__main__':
    main()



import argparse
import os
from make_data import manage_path_argument

''' 
Main Script of Structural Analysis

parameters: source_folder (of deconvolted images)


Execute:

o script esegue:
 - la fase ALFA (equalizzazione e segmentazione delle immagini, salvando i formati indicati in parameters.txt)
 - la fase GAMMA (analisi orientazione e salvataggio risultati in tabella R.npy)
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

    # create folder path of segmented images
    segmented_path = os.path.join(base_path, 'segmented', stack_name)

    # call  ALFA_volume_analysis script
    os.system('python3 ALFA_volume_analysis.py -sf {}'.format(source_path))

    # call  ALFA_volume_analysis script
    os.system('python3 GAMMA_orientation_analysis_no_outlier.py -sf {}'.format(segmented_path))


def main():

    # define parser
    parser = argparse.ArgumentParser(
        description='Strip images enhanced and 3D Structural analysis',
        epilog='Author: Francesco Giardini <giardini@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # command line REQUIRED argument
    parser.add_argument('-sf', '--source_folder', nargs='+', required=True, help='input images path')

    # run analyis
    structural_analysis(parser)

if __name__ == '__main__':
    main()



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
    print()

    args = parser.parse_args()

    # extract path string
    source_path = manage_path_argument(args.source_folder)
    source_path = source_path.replace(' ', '\ ')  # correct whitespace with backslash

    # extract preference about outlier remotion in GAMMA
    _no_outlier_remotion = args.no_outlier_remotion  # dafault: False -> outlier remotion is executed

    # extract base path and stack name
    base_path = os.path.dirname(os.path.dirname(source_path))
    stack_name = os.path.basename(source_path)

    # create folder path of segmented images
    segmented_path = os.path.join(base_path, 'segmented', stack_name)

    print('Running Strip Analysis on: \n', base_path)
    print('Selecting No_Outlier_Remotion = {}'.format(_no_outlier_remotion))
    print(' *********************************************************************************')
    print('\n')

    # call  ALFA_volume_analysis script
    os.system('python3 ALFA_volume_analysis.py -sf {}'.format(source_path))

    # call  GAMMA_orientation_analysis script
    if _no_outlier_remotion is False:
        os.system('python3 GAMMA_orientation_analysis.py -sf {}'.format(segmented_path))  # default
    else:
        # os.system('python3 GAMMA_orientation_analysis_no_outlier_DEPREC.py -sf {}'.format(segmented_path))
        os.system('python3 GAMMA_orientation_analysis.py -sf {} -nor'.format(segmented_path))  # NO OUTLIER REMOTION

def main():

    # define parser
    parser = argparse.ArgumentParser(
        description='Strip images enhanced and 3D Structural analysis',
        epilog='Author: Francesco Giardini <giardini@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # command line REQUIRED argument
    parser.add_argument('-sf', '--source_folder', nargs='+', required=True, help='input images path')
    parser.add_argument('-nor', action='store_true', default=False, dest='no_outlier_remotion',
                        help='Add \'-nor\' if you don\'t want to execute the outlier remotion before estimate '
                             'statistics on R inside GAMMA_Analysis.py. '
                             'Threshold ìs evaluated on on (Ycomp/PSDratio) with Hyperbole function.'
                             'Default: False -> Outlier remotion is executed.')

    # run analyis
    structural_analysis(parser)

if __name__ == '__main__':
    main()



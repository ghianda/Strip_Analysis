import argparse
import os

from custom_tool_kit import manage_path_argument

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def myocardial_fraction_recursive(parser):

    print('\n\n')
    print('*********************************************************************************')
    print('******************* Myocardial_fraction_recursive.py script *********************')
    print('*********************************************************************************')

    args = parser.parse_args()

    # extract path string (folder where are all the strips to analyze)
    global_path = manage_path_argument(args.source_folder)
    print('global_path: ', global_path)

    samples_folders = [f for f in os.listdir(global_path)]
    print('number of samples finded: ', len(samples_folders), '\n')

    # analyze every samples
    for f in samples_folders:
        sample_path = os.path.join(global_path, f)
        print(bcolors.WARNING + '====> Analyzing {}'.format(sample_path) + bcolors.ENDC)

        # start analysis script
        os.system('python3 myocardial_fraction.py -sf {} -y1 0 -y2 0'.
                  format(os.path.join(sample_path, 'mask_bin', 'stitched_stack')))


if __name__ == '__main__':

    # define parser
    parser = argparse.ArgumentParser(
        description='Script for Myocardial fraction analysis on a set of samples',
        epilog='Author: Francesco Giardini <giardini@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # command line of the REQUIRED argument
    parser.add_argument('-sf', '--source_folder', nargs='+', required=True,
                        help='path of the folder containing all the samples to analyze')

    # run script
    myocardial_fraction_recursive(parser)

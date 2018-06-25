import numpy as np
import argparse

from skimage import exposure

from custom_image_tool import save_tiff, normalize, create_img_name_from_index
from custom_tool_kit import nextpow2
from make_data import make_dataset, create_stack, stack_to_array


def CLAHE(parsers):
    print('START')
    args = parser.parse_args()
    if args.source_file_list is None or args.output is None:
        print('You should select the Path of Source Images and destination folder.')
        parser.print_help()

    # extract images
    if args.unique_file_tif in ['1', 'true', 'True', 't', 'T', 'y', 'Y', 'yes', 'Yes']:
        imgs = stack_to_array(args.source_file_list)  # [height, width, slices]
    else:
        source_data = make_dataset(args.source_file_list)
        imgs = create_stack(source_data)

    # process
    if imgs is not None:

        (weight, height, num_of_slices) = imgs.shape
        ksize = nextpow2(weight / 8)
        clip = 0.08

        print('write in to {} \n'.format(args.output))

        for z in range(imgs.shape[2]):
            img = normalize(imgs[:, :, z])
            img_name = create_img_name_from_index(z, post="_clahe")

            img_eq = normalize(exposure.equalize_adapthist(image=img, kernel_size=ksize, clip_limit=clip))

            save_tiff(img=img_eq, img_name=img_name, prefix='', comment='', folder_path=args.output)
            print(img_name)
        print(' \n ** Process Finished \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLAHE on source path images')
    parser.add_argument('-sf', '--source-file-list', nargs='+', help='Images to segment', required=True)
    parser.add_argument('-o', '--output', help='Destination Path', required=True)
    parser.add_argument('-u', '--unique-file-tif', nargs='+',
                            help='True (or 1) if file is unique tiff stack, False (or 0) if it is a directory', required=False)

    CLAHE(parser)

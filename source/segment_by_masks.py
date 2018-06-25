import numpy as np
import argparse

from skimage import exposure

from custom_image_tool import save_tiff, normalize
from make_data import make_dataset, create_stack

"""
    Do AND between source tif image and mask tif image (binary image), and saved results in tif

    Executable from terminal
    parameter:
     - source path
     - mask path
     - output path 
     
     example
     
     python3 segment_by_masks -sf /home/.../original_images -mf /home/.../mask_images -o /home/.../segmented_images
"""


def segment_by_masks(parsers):
    print('START')
    args = parser.parse_args()
    if args.source_file_list == None or args.mask_file_list == None:
        print('You should select the Path of Source Images and masks Images.')
        parser.print_help()

    source_data = make_dataset(args.source_file_list)
    print('source_file_list ', args.source_file_list)
    print('source_data.len() = ', len(source_data))

    mask_data = make_dataset(args.mask_file_list)
    print('mask_file_list ', args.mask_file_list)
    print('mask_data.len() = ', len(mask_data))

    dest_folder = args.output
    print('dest_folder ', dest_folder)

    imgs = create_stack(source_data)
    print('imgs.shape[2] = ', imgs.shape[2])

    masks = create_stack(mask_data)
    print('imgs.shape[2] = ', imgs.shape[2])

    if imgs is not None and masks is not None:

        if imgs.shape[2] != masks.shape[2]:
            print('Warning! Dimension of Images and masks mismatch.')

        else:
            # todo script
            print('write in {} \n'.format(dest_folder))

            segmented = np.zeros(imgs.shape)
            print('\n segmented shape: ', segmented.shape)
            print('imgs shape: ', imgs.shape)
            print('masks shape: ', masks.shape)

            for z in range(imgs.shape[2]):
                img = normalize(imgs[:, :, z])
                img_eq = exposure.equalize_adapthist(img, clip_limit=0.03)

                segmented[:, :, z] = img_eq * masks[:, :, z]

                save_tiff(img=segmented[:, :, z],
                          img_name=str(z), prefix='s',
                          comment='dec_seg',
                          folder_path=dest_folder)
                print('img n° {}'.format(z))

            print(' \n ** Proces Finished \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEST PARSE FILE')
    parser.add_argument('-sf', '--source-file-list', nargs='+', help='Images to segment', required=False)
    parser.add_argument('-mf', '--mask-file-list', nargs='+', help='Masks for segmentation', required=False)
    parser.add_argument('-o', '--output', help='Destination Path', required=True)

    segment_by_masks(parser)

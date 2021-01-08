import numpy as np
import os

# l'ho fattao al volo hardcoded:
path = '/home/francesco/JupyterProjects/Strip/2.0_seven_strips_donor/data'
g = 3  # grana disarray

# read list of sub directories (only folders):
dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
print(dirs)

D = list() # list of disarray matrices
for d in dirs:
    current_npy_path = os.path.join(path, d, 'Disarray_matrix_g{0}x{0}x{0}_of_orientation_Results.npy'.format(g))
    print('Loading from:')
    print(current_npy_path)
    D.append(np.load(current_npy_path))

# estimate averaged disarray and write in a txt file
txt_filepath = os.path.join(path, 'Averaged_Disarray_g{0}x{0}x{0}.txt'.format(g))

with open(txt_filepath, 'w') as txt:
    for (i, d) in enumerate(D):
        dv = d[d != -1]  # extract only valid values
        print('*** i = {} ***'.format(i))
        print('dv.shape: ', dv.shape)
        print('dv.mean : ', dv.mean())
        txt.write('* Sample: {}: \n'.format(dirs[i]))
        txt.write('- d.shape: {} \n'.format(str(d.shape)))
        txt.write('- valid values: {} \n'.format(dv.shape[0]))
        txt.write('- Disarray AVG: {0:0.3f}% \n'.format(dv.mean()))
        txt.write('- Disarray STD: {0:0.3f}% \n'.format(dv.std()))
        txt.write('\n\n')
        print('\n')
# l'ho fatto al volo

import numpy as np

# load list of disarray matrices:
D = list()

for i in range(1, 8):
    D.append(np.load('/home/francesco/JupyterProjects/Strip/2.0_seven_strips_donor/orientation_matrices/Disarray_matrix_g4x4x4_of_R_{}.npy'.format(i)))

results = list()
for (i, d) in enumerate(D):
    dv = d[d != -1]
    print('*** i = {} ***'.format(i))
    print('dv.shape: ', dv.shape)
    print('dv.mean : ', dv.mean())

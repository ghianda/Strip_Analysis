# constant and parameters for the groundtruth generation
# from [DELL] -> JupyterProjects/Strip/accuracy_orientation/single_block/single_block_analysis_accuracy_STRIP_heatmaps_articolo_BOCCHI.ipynb


#########################    INPUT    #############################
# settings = {}
#
# # list of rotation to apply
# theta_list = list(range(-30, 35, 5))
# phi_list   = list(range(-30, 35, 5))
#
# # theta_list = list([-45, -30, -15, 0, 15, 30, 45])
# # phi_list   = list([-60, -30, 0, 30, 60])
#
#
# _smooth_z      = False
# mode           = 'reflect'  # {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional, (REFLECT)
#
# _save_numpys   = False
# _display_plot  = False
# _save_plot     = False  # vectors plot!!
# plt_folder     = 'vectors_plots'  # it will be automatically created
# _save_heatmap  = True
# maps_folder    = 'error_maps_smooth'
# _save_tiffs    = False
# tiff_folder    = 'block_rotated'
#
# _save_lin_plot     = True
# theta_lineplt_name = 'theta_estimation_no_smooth.pdf'
# capabiliy_plt_name = 'resolution_capability_no_smooth.pdf'

###################################################################

settings = {}

# list of rotation to apply
settings['theta_list'] = list(range(-30, 35, 5))
settings['phi_list']   = list(range(-30, 35, 5))

# theta_list = list([-45, -30, -15, 0, 15, 30, 45])
# phi_list   = list([-60, -30, 0, 30, 60])


settings['_smooth_z']      = False
settings['mode']           = 'reflect'  # {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional, (REFLECT)

settings['_save_numpys']   = False
settings['_display_plot']  = False
settings['_save_plot']     = False  # vectors plot!!
settings['plt_folder']     = 'vectors_plots'  # it will be automatically created
settings['_save_heatmap']  = True
settings['maps_folder']    = 'error_maps_smooth'
settings['_save_tiffs']    = False
settings['tiff_folder']    = 'block_rotated'

settings['_save_lin_plot']     = True
settings['theta_lineplt_name'] = 'theta_estimation_no_smooth.pdf'
settings['capabiliy_plt_name'] = 'resolution_capability_no_smooth.pdf'

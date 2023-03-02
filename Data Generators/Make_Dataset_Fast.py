from simple_cross.Multi_X_Gen import simp_generator
from simple_cross.Multi_X_Sim import simp_simulator
from Blank_Data_Image_Generator import generate_blanks

#%% - User Inputs
xdim = 88
ydim = 128
time_dimension = 100
sig_pts = 28
dats_set_title = "Testing multi data gen"
data_sets_folder =r"C:\Users\Student\Documents\UNI\Onedrive - University of Bristol\Yr 3 Project\Circular and Spherical Dummy Datasets\\"
output_dir = data_sets_folder + dats_set_title + "\Data\\"

number_of_blanks = 2
number_single_x = 2
number_of_two_x = 2
number_of_three_x = 2
number_of_four_x = 2

shift_x_positions = False
rotate_x_positions = True


#%% - Compute
# Gen Blanks
generate_blanks(xdim=xdim, ydim=ydim, number_of_files=number_of_blanks, output_dir=output_dir)

# Gen Signals  - can clean up all the intro code to one line if sort out args slightly (needs number of images removed and then no ratios to types of images just straiht values, and finally the shift and rotate params dshould accept boolean instead of 0-1 for user ease of following code)
total_x_images = number_single_x + number_of_two_x + number_of_three_x + number_of_four_x
datasplit_ratios = [number_single_x/total_x_images, number_of_two_x/total_x_images, number_of_three_x/total_x_images, number_of_four_x/total_x_images]

if shift_x_positions:
    shift = 1
else:
    shift = 0

if rotate_x_positions:
    rotate = 1
else:
    rotate = 0

simp_generator(datasplit_ratios, total_x_images, sig_pts, xdim, ydim, time_dimension, shift)



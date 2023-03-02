from Simple_Cross.Multi_X_Gen import simp_generator
from Blank_Data_Image_Generator import generate_blanks

#%% - User Inputs
xdim = 88
ydim = 128
time_dimension = 100
sig_pts = 28
output_dir = ###

number_of_blanks = 10
number_single_x = 
number_of_two_x = 
number_of_three_x = 
number_of_four_x = 

shift_x_positions = True
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
simp_generator(datasplit_ratios, total_x_images, sig_pts, xdim, ydim, time_dimension, shift, rotate)



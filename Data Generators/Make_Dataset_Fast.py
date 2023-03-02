from Simple_Cross import #####function for x gen    basic_Xgen, multiX_gen, x_gen_v2
from Blank_Data_Image_Generator import generate_blanks

xdim = 88
ydim = 128
time_dimension = 100
output_dir = ###
number_of_blanks = 10

#x_gen_V2(arg1, arg2, arg3, arg4, argn...)
generate_blanks(xdim=xdim, ydim=ydim, number_of_files=number_of_blanks, output_dir=output_dir)


import os
import shutil
from DC3D_V3.Autoencoders.Dataset_Integrity_Check_V1 import dataset_integrity_check

def parse_module_name(string):
    # Protection from module being in subfolder so _get_module retuns subfolder.module_name we want just module name, so following lines check for any dots in the module name and if so find the right most dot and returns the string following it, otheriwse just retunrs it's input 
    dot_index = string.rfind('.')
    if dot_index != -1:
        left_string = string[:dot_index].replace('.', '/')     # Text to left of right most dot (this is the subfolder(s) names if there is) the replace replaces any dots with / for proper folder naming
        right_string = string[dot_index+1:]  # Text to right of right most dot (this is module name)
        return right_string, left_string # Module name, subfolder
    else:
        return string, None # Module name, None (there is no subfolder in this case)



# Find autoencoder module name
module_name = dataset_integrity_check.__module__
module_name, subfolder = parse_module_name(module_name)

# Define the filename to search for ()
filename = module_name + '.py'


search_dir = os.getcwd()  # current working directory
if subfolder != None:
    search_dir = search_dir + "/" + subfolder
    print("subfolder")
print("search loc", search_dir)
output_dir = search_dir + '/location/'
# Traverse the directory tree to find the file
file_path = None
for root, dirs, files in os.walk(search_dir):
    if filename in files:
        file_path = os.path.join(root, filename)
        break

# Copy the file to a new location with modified filename
if file_path is None:
    print(f"File '{filename}' not found in '{search_dir}' or its subdirectories")
else:
    new_filename = os.path.splitext(filename)[0] + "-TEST" + os.path.splitext(filename)[1]
    os.makedirs(output_dir, exist_ok=True)
    new_file_path = os.path.join(output_dir, new_filename)
    shutil.copyfile(file_path, new_file_path)
    print(f"File '{filename}' copied to '{new_file_path}'")

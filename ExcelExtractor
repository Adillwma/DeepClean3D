import pandas as pd
import os

# set the directory path where the txt files are located
directory = "C:\\Users\\maxsc\\OneDrive - University of Bristol\\Adill Al-Ashgar - Adill Al-Ashgars files\\Yr 3 Project Results\\"

# create an empty list to store the extracted data
data = []

# iterate through all txt files in the directory and subdirectories
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.txt'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    # read the contents of the txt file
                    contents = f.read()
                    
                    # split the contents into lines
                    lines = contents.split('\n')
                    
                    # create an empty dictionary to store the key-value pairs
                    kv_pairs = {}
                    
                    # iterate through each line of the txt file
                    for line in lines:
                        # check if the line contains a key-value pair
                        if line and line[0].isalpha() and ':' in line:
                        #if ':' in line:
                            # split the line into key and value
                            key, value = line.split(':', 1)
                            
                            # add the key-value pair to the dictionary
                            kv_pairs[key.strip()] = value.strip()
                    
                    # append the dictionary to the data list
                    data.append(kv_pairs)
            except:
                pass

# convert the data list into a pandas DataFrame
df = pd.DataFrame(data)

# set the output path for the Excel file
output_path = "C:\\Users\\maxsc\\OneDrive - University of Bristol\\3rd Year Physics\\Project\\ExcelModels\\ExtractedExcel2.xlsx"

# save the DataFrame to an Excel file
df.to_excel(output_path, index=False)
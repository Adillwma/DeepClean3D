# File not found error already returns well defined error message so not including reporting for it

# Scrape Training Time
def get_training_time(file_path):
    with open(file_path, 'r') as f:                                   # Opens the file at the given file path using a with to ensure proper file handline (makes sure file is closed properly)
        for line in f:                                                # Iterates through file line by line
            if 'Training Time:' in line:                              # For each line, checks if the string "Training Time:" is present
                time_str = line.strip().split(': ')[1].split(' ')[0]  # If it is, it extracts the training time string, which is the value after the colon and before the first space character
                return float(time_str)                                # Converts the extracted string to a float and returns it
    print('Training Time not found in file')               # If the function reaches the end of the file without finding the training time string, it warns user but does not break for error, the function will return a flag so main body program can implement its own conditions if neded.
    return ("Error")
"""
# Test Driver
training_time = get_training_time('path/to/file.txt')
print(training_time)
"""
import os
import matplotlib.pyplot as plt

# GET THE TRAINING TIME FOR EACH MODEL! THAT WAY FULL TRAIN TIME FOR ANCESTRY CAN BE CALULATED TOO 

# Settings
results_folder_path =  "N:/DeepClean3D Project Folder/Yr 3 Project Results/"  # Path to results folders


# Program
print("Scanning results folders for model family tree data...")

# Scan each folder in the results folder for a txt file 
model_ids = []
filenames = {}
parent_model_ids = {}

for folder_name in os.listdir(results_folder_path):
    folder_path = os.path.join(results_folder_path, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith("Network Summary.txt"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.startswith("Model_ID:"):
                            model_id = line.split(":")[1].strip()
                            print(f"Folder: {folder_name}, Model ID: {model_id}")
                            model_ids.append(model_id)
                            filenames[model_id] = file_path

                        if line.startswith("parent_model_id:"):
                            parent_model_id = line.split(":")[1].strip()
                            print(f"Folder: {folder_name}, Parent Model ID: {parent_model_id}")
                            parent_model_ids[model_id] = parent_model_id
                            break  # No need to read further lines once we found the parent id. model id comes first so is safe to check for first each line and leave brbeak condition onn finding the parent id.
                
                break  # No need to read further files once we found the txt file.

# Create a family tree structure displaying parent-child relationships
family_tree = {}
for model_id in model_ids:
    parent_id = parent_model_ids[model_id]
    if parent_id not in family_tree:
        family_tree[parent_id] = []
    family_tree[parent_id].append(model_id)
# Function to recursively print the family tree
def print_family_tree(model_id, level=0):
    indent = "  " * level
    print(f"{indent}- Model ID: {model_id}")
    if model_id in family_tree:
        for child_id in family_tree[model_id]:
            print_family_tree(child_id, level + 1)

def plot_family_tree(model_id, x=0, y=0, level=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Model Family Tree")
        ax.axis('off')

    ax.text(x, y, f"Model ID: {model_id}", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))

    if model_id in family_tree:
        num_children = len(family_tree[model_id])
        for i, child_id in enumerate(family_tree[model_id]):
            child_x = x + (i - (num_children - 1) / 2) * 4
            child_y = y - 2
            ax.plot([x, child_x], [y - 0.2, child_y + 0.2], 'k-')
            plot_family_tree(child_id, child_x, child_y, level + 1, ax)

    if level == 0:
        plt.show()

found = len(model_ids)
print(f"Scan complete. Found {found} model IDs and parent relationships.")

if found < 0:
    # Print the family tree starting from foundation models (those without parents)
    print("Model Family Tree:")
    for root_model_id in family_tree.get(None, []):
        print_family_tree(root_model_id)

    for root_model_id in family_tree.get(None, []):
        plot_family_tree(root_model_id) 
        

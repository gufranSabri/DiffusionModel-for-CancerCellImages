import os
import random
import shutil

def move_random_files(source_folder, destination_folder, percentage):
    all_files = os.listdir(source_folder)
    number_of_files_to_move = int(len(all_files) * percentage)

    selected_files = random.sample(all_files, min(number_of_files_to_move, len(all_files)))

    for file_name in selected_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)
        print(f"Moved: {file_name}")

if __name__ == "__main__":
    train_all_folder = "/Users/gufran/Developer/Projects/AI/CancerDM/data/train/all"
    train_hem_folder = "/Users/gufran/Developer/Projects/AI/CancerDM/data/train/hem"
    test_all_folder = "/Users/gufran/Developer/Projects/AI/CancerDM/data/val/all"
    test_hem_folder = "/Users/gufran/Developer/Projects/AI/CancerDM/data/val/hem"
    percentage_to_move = 0.2

    move_random_files(train_all_folder, test_all_folder, percentage_to_move)
    move_random_files(train_hem_folder, test_hem_folder, percentage_to_move)

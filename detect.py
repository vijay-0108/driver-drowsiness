import os
import cv2
import matplotlib.pyplot as plt

def check_images(dataset_path):
    """
    Checks for missing or corrupted image files in the dataset.
    """
    total_files = 0
    corrupted_files = 0
    corrupted_file_paths = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_files += 1
            img = cv2.imread(file_path)  # Try to read the image
            if img is None:  # If the image is unreadable
                print(f"Corrupted or unreadable file: {file_path}")
                corrupted_files += 1
                corrupted_file_paths.append(file_path)
    
    print(f"\nTotal files checked: {total_files}")
    print(f"Corrupted files found: {corrupted_files}")
    return corrupted_file_paths

def check_file_formats(dataset_path, valid_extensions=(".jpg", ".jpeg", ".png")):
    """
    Checks if all files in the dataset have valid image extensions.
    """
    invalid_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                invalid_files.append(os.path.join(root, file))
                print(f"Invalid file format: {os.path.join(root, file)}")
    print(f"\nInvalid files found: {len(invalid_files)}")
    return invalid_files

def analyze_class_distribution(dataset_path):
    """
    Analyzes and plots the distribution of classes in the dataset.
    """
    class_counts = {}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))

    # Print and plot the distribution
    print(f"\nClass Distribution: {class_counts}")
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.show()

# Example usage
dataset_path = r'C:\Users\saich\Downloads\Drowsiness detection\dataset_new\train'

print("=== Checking for Missing or Corrupted Files ===")
corrupted_files = check_images(dataset_path)

print("\n=== Checking for Invalid File Formats ===")
invalid_files = check_file_formats(dataset_path)

print("\n=== Analyzing Class Distribution ===")
analyze_class_distribution(dataset_path)

# Optional: Remove corrupted or invalid files
remove_files = input("\nDo you want to remove corrupted and invalid files? (y/n): ")
if remove_files.lower() == 'y':
    for file_path in corrupted_files + invalid_files:
        os.remove(file_path)
        print(f"Removed: {file_path}")
    print("All corrupted and invalid files have been removed.")

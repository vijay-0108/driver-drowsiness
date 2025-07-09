import os
import matplotlib.pyplot as plt

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
analyze_class_distribution(dataset_path)
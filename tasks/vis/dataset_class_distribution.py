
import glob
import os
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.datasets.datasets import get_dataset_class_by_name
from src.datasets.datasets import get_dataset_class_by_name


@hydra.main(version_base=None, config_path='../../conf', config_name='dataset_class_distribution')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')

    dataset_name = cfg_dict['dataset']['name']
    dataset_class = get_dataset_class_by_name(dataset_name)
    dataset = dataset_class(split='train')

    plot_class_distribution(dataset_name, dataset.images_dir)


def plot_class_distribution(dataset_name, dataset_path):
    # Get all subdirectories (classes) in the dataset path
    class_directories = glob.glob(os.path.join(dataset_path, "*"))

    # Count the number of images in each class
    class_counts = []
    class_labels = []
    for class_dir in class_directories:
        class_label = os.path.basename(class_dir)
        image_files = glob.glob(os.path.join(class_dir, "*"))  # Change the extension as per your dataset
        class_count = len(image_files)
        class_labels.append(class_label)
        class_counts.append(class_count)

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.bar(class_labels, class_counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=90)

    # Save the figure as a PDF file
    output_file = os.path.join(f'{dataset_name}_class_distribution.pdf')
    plt.savefig(output_file, format='pdf')
    plt.close()


if __name__ == '__main__':
    main()


import random
import numpy as np
import matplotlib.pyplot as plt


def plot_per_class_samples(dataset, n=3):
    _, plots = plt.subplots(n, len(dataset.labels), figsize=(20,10))

    for label in dataset.labels:
        data_indices = np.where(dataset.images_data['labels'] == label)

        for i in range(n):
            data_index = random.choice(data_indices)            
            data_point = dataset[data_index]

            plot = plots[i][label]

            image = data_point['image'].cpu().numpy()
            image = np.moveaxis(image, 0, -1)
            image -= image.min()
            image /= image.max()

            plot.set_title(dataset.labels_text[data_point['label']])
            plot.axis('off')
            plot.imshow(image)

    plt.show()

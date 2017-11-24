import random
import time
import matplotlib.pyplot as plt
import numpy as np
import cifar10
from lab8 import predict, train_Batch, np


def plot_9images(images, cls_idx_true, cls_idx_pred=None, all_cls_names=None, smooth=True):
    assert len(images) == len(cls_idx_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_idx_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :],
                  interpolation=interpolation)

        # Name of the true class.
        cls_true_name = all_cls_names[cls_idx_true[i]]

        # Show true and predicted classes.
        if cls_idx_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = all_cls_names[cls_idx_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def main():
    class_names = cifar10.load_class_names()
    images_train, cls_idx_train, labels_train = cifar10.load_training_data()
    images_test, cls_idx_test, labels_test = cifar10.load_test_data()
    
    images_test = np.reshape(images_test, (images_test.shape[0], 32 * 32, 3))[:,:,0]
    images_test = np.reshape(images_test, (images_test.shape[0], 32, 32))
    
    images_train = np.reshape(images_train, (images_train.shape[0], 32 * 32, 3))[:,:,0]
    images_train = np.reshape(images_train, (images_train.shape[0], 32, 32))
    
    # Plot the first 9 training images and labels
    plot_9images(images=images_train[0:9], cls_idx_true=cls_idx_train[0:9],
                all_cls_names=class_names, smooth=True)

    # Build your predictor
    NN = train_Batch(images_train, labels_train)

    # Visualize your prediction
    samples = random.sample(range(len(images_test)), 9)
    plot_9images(images=images_test[samples], cls_idx_true=cls_idx_test[samples],
                 cls_idx_pred=predict(images_test[samples], NN), all_cls_names=class_names, smooth=True)

    print(f'\nAccuracy: {(predict(images_test, NN) == cls_idx_test).mean() * 100}%\n')

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)

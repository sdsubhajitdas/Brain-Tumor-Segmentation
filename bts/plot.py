import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def result(image, mask, output,transparency=0.35,save_path=None):
    """This method plots the original image,activation map of the bottleneck layer
    and reconstructed image.

    Keyword Arguments:
    original   - Numpy array of the original data (shape = (784,))
    activation - Numpy array of the activation map data (shape = (36,))
    created    -  Numpy array of the reconstructed data (shape = (784,))

    """

    plt.figure(figsize=(15, 12))

    gs = gridspec.GridSpec(4, 8)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[0:2, 0:4])
    ax2 = plt.subplot(gs[0:2, 4:8])
    ax3 = plt.subplot(gs[2:4, 0:4])
    ax4 = plt.subplot(gs[2:4, 4:8])

    ax1.set_title("Original Mask", fontdict={'fontsize': 16})
    ax1.imshow(mask, cmap='gray')
    ax1.set_axis_off()

    ax2.set_title("Constructed Mask", fontdict={'fontsize': 16})
    ax2.imshow(output, cmap='gray')
    ax2.set_axis_off()

    ax3.set_title("Original Image", fontdict={'fontsize': 16})
    ax3.imshow(image, cmap='gray')
    ax3.set_axis_off()
    
    seg_output = output*transparency
    seg_image = np.add(image,seg_output)/2
    ax4.set_title("Segmented Image", fontdict={'fontsize': 16})
    ax4.imshow(seg_image, cmap='gray')
    ax4.set_axis_off()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def loss_graph(loss_list, save_plot=None):
    plt.figure(figsize=(20, 10))
    plt.title('Loss Function Over Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    line = plt.plot(loss_list, marker='o')
    plt.legend((line), ('Loss Value',), loc=1)
    if save_plot:
        plt.savefig(save_plot)
    plt.show()

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def show_images(image_batch,bs,gray=False):
    """

    :param image_batch: ndarray BHWC or BHW1
    :param bs:
    :param gray: 若img为HW1的Gray图，则置True
    :return:
    """
    columns = 4
    rows = (bs + 1) // (columns)
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        if gray:
            RGB=cv2.cvtColor(image_batch[j],cv2.COLOR_GRAY2RGB)
            plt.imshow(RGB)
        else:
            plt.imshow(image_batch[j])
    plt.show()
import matplotlib.pyplot as plt
import torchvision
import matplotlib.gridspec as gridspec


def show(image, segment, depth):
    plt.figure(figsize = (10,10))
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(wspace=0, hspace=0)
    plt.axis('on')

    ax1 = plt.subplot(0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    ax1.imshow(torchvision.utils.make_grid(image).permute(1, 2, 0))

    ax2 = plt.subplot(1)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_aspect('equal')
    ax2.imshow(torchvision.utils.make_grid(segment).permute(1, 2, 0))

    ax3 = plt.subplot(2)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_aspect('equal')
    ax3.imshow(torchvision.utils.make_grid(depth).permute(1, 2, 0))


    plt.show()

    
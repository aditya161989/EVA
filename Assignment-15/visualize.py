import matplotlib.pyplot as plt
import torchvision
import numpy as np
import skimage
def show(output_tensors, original_tensors, epoch = None, fig_size=(10,10), *args, **kwargs):
    
    allimage=[]
    for i in range(20):
        img_set = np.vstack([torchvision.utils.make_grid(original_tensors[i]).permute(1, 2, 0), torchvision.utils.make_grid(output_tensors[i]).permute(1, 2, 0)])
        allimage.append(img_set)
    allimage=np.stack(allimage)
    vm=skimage.util.montage(allimage, multichannel=True, fill=(0,0,0),grid_shape=(4,5))
    plt.figure(figsize=(15, 20))
    plt.imshow(vm)
    plt.axis('off')
    plt.grid('off')
#     size = 10
#     plt.figure(figsize = (10,10))
#     if output_tensors.shape[0] < size:
#         size = output_tensors.shape[0]
#     gs1 = gridspec.GridSpec(1, size)
#     gs1.update(wspace=0, hspace=0) # set the spacing between axes. 

#     for i in range(size):
#    # i = i + 1 # grid spec indexes from 0
#         ax1 = plt.subplot(gs1[i])
#         plt.axis('on')
#         ax1.set_xticklabels([])
#         ax1.set_yticklabels([])
#         ax1.set_aspect('equal')
#         ax1.imshow(torchvision.utils.make_grid(original_tensors[i]).permute(1, 2, 0))
#     plt.suptitle('Epoch : {} -- Original Images'.format(epoch))
#     plt.show()

#     plt.figure(figsize = (10,10))

#     gs1 = gridspec.GridSpec(1, size)
#     gs1.update(wspace=0, hspace=0) # set the spacing between axes. 

#     for i in range(size):
#    # i = i + 1 # grid spec indexes from 0
#         ax1 = plt.subplot(gs1[i])
#         plt.axis('on')
#         ax1.set_xticklabels([])
#         ax1.set_yticklabels([])
#         ax1.set_aspect('equal')
#         ax1.imshow(torchvision.utils.make_grid(output_tensors[i]).permute(1, 2, 0))
#     plt.suptitle('Epoch : {} -- Predicted Images'.format(epoch))
#     plt.show()
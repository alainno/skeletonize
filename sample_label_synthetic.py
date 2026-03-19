import os
import cv2
import matplotlib.pyplot as plt

#
cur_dir = os.path.dirname(os.path.realpath(__file__))
imgs_dir = os.path.join(cur_dir, 'datasets/simulated/train/images')
lbls_dir = os.path.join(cur_dir, 'datasets/simulated/train/masks')

fig = plt.figure(figsize=(4*3,4))
fig.subplots_adjust(wspace=0.1)

img_names = os.listdir(imgs_dir)
img_path = os.path.join(imgs_dir, img_names[0])
if os.path.isfile(img_path):
    img = cv2.imread(img_path, 0)
    _,bin_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # skeleton = cv2.ximgproc.thinning(bin_img, cv2.ximgproc.THINNING_GUOHALL)
    skeleton = cv2.imread(os.path.join(lbls_dir, img_names[0]), 0)
    
    plt.subplot(1,3,1), plt.axis('off'), plt.imshow(img, cmap='gray')
    plt.subplot(1,3,2), plt.axis('off'), plt.imshow(bin_img, cmap='gray')
    plt.subplot(1,3,3), plt.axis('off'), plt.imshow(skeleton, cmap='gray')
    
plt.savefig('outputs/sample-label-synthetic.png', bbox_inches='tight', pad_inches=0)
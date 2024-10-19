import glob
import os
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, binary_closing
from skimage.filters import threshold_otsu
from skimage import img_as_uint
import cv2

def emptyDir(folder):
    '''
    Eliminar archivos contenidos en 'folder'
    '''
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def initDirectory(path):
    if(os.path.isdir(path)):
        emptyDir(path)
    else:
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)

def save_skeleton(src, dst):
    bin_img = cv2.imread(src, 0)
    # dm = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
    # dm = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX)
    skeleton = skeletonize(bin_img)
    skeleton = binary_closing(skeleton)
    skeleton = img_as_uint(skeleton)
    return cv2.imwrite(dst, skeleton)
    # skeleton.save(dst, 'PNG')
    # return True

def skeletonize_batch(src_dir, dst_dir):
    for filename in glob.glob(os.path.join(src_dir,"*.png")):
        print('filename:',filename)
        if not save_skeleton(filename, os.path.join(dst_dir, os.path.basename(filename))):
            return False
    return True

if __name__ == "__main__":
    print("Generando skeleton")
    # read dataset images and create skeleton
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(cur_dir, '../datasets/ofda/train/images')
    gt_path = os.path.join(cur_dir, '../datasets/ofda/train/masks')

    initDirectory(gt_path)
    
    source_dir = os.path.join(cur_dir, '../../dmnet/datasets/ofda/ofda_segmentation')
    print('source_dir',source_dir)
    target_dir = gt_path
    if not skeletonize_batch(source_dir, target_dir):
        print("Error")
    
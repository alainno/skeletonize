import unittest
import cv2
import h5py
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

from skimage import img_as_uint, io as ioo
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, binary_closing

class Skeleton:
    # def __init__(self, src_dir):
    #     self.src_dir = src_dir
        
    def save(self, src, dst):
        img = rgb2gray(imread(src))
        thresh = threshold_otsu(img)
        binary = img > thresh
        skeleton = skeletonize(binary)
        skeleton = binary_closing(skeleton)
        # skeleton = img_as_uint(skeleton)
        skeleton = skeleton.astype(np.uint8) * 255
        try:
            ioo.imsave(fname=dst, arr=skeleton)    
            # return cv2.imwrite(dst, dm_img)
            return True
        except:
            return False
    
    def create_batch(self, src_dir, dst_dir, seg_dir):
        print(src_dir)
        for filename in glob.glob(os.path.join(src_dir,"*.png")):
            basename = os.path.basename(filename)
            #[prefix,id] = basename.split("_")
            #print(prefix, id)
            #source = os.path.join(seg_dir, id)
            source = os.path.join(seg_dir, basename)
            #print(source)
            if not self.save(source, os.path.join(dst_dir, basename)):
                return False
        return True
    
    #def show_distance_maps(self):
        

class TestSkeletons(unittest.TestCase):
    # def test_one(self):
    #     cur_dir = os.path.dirname(os.path.realpath(__file__))
    #     base_path = os.path.join(cur_dir, '..', 'datasets', 'ofda')
    #     filename = "g1_0157.png"
    #     [prefix,id] = filename.split("_")
    #     print(prefix, id)
    #     source = os.path.join(base_path, 'cropped', 'cropped', 'exported', 'g1', id)
    #     target = os.path.join(base_path, 'train', 'masks', f'{prefix}_{id}')
    #     skel = Skeleton()
    #     self.assertEqual(skel.save(source, target), True, "generación incorrecta")
        
    def test_batch(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.join(cur_dir, '..', 'datasets', 'ofda')
        src_dir = os.path.join(base_path, 'train', 'images')
        dst_dir = os.path.join(base_path, 'train', 'masks')
        seg_dir = os.path.join(base_path, 'train', 'segments')
        skel = Skeleton()
        self.assertEqual(skel.create_batch(src_dir, dst_dir, seg_dir), True, "creación satisfactoria")

    #def test_batch(self):
    #    cur_dir = os.path.dirname(os.path.realpath(__file__))
    #    base_path = os.path.join(cur_dir, '..', 'datasets', 'ofda')
    #    src_dir = os.path.join(base_path, 'test', 'images')
    #    dst_dir = os.path.join(base_path, 'test', 'masks')
    #    seg_dir = os.path.join(base_path, 'cropped', 'cropped', 'exported', 'g1')
    #    skel = Skeleton()
    #    self.assertEqual(skel.create_batch(src_dir, dst_dir, seg_dir), True, "creación satisfactoria")
        
if __name__ == "__main__":
    unittest.main()
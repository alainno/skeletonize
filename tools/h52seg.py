import h5py
import cv2
import numpy as np
import os
import glob

def dm2seg(filepath, output):
    with h5py.File(filepath, "r") as hf:
        distance_map = hf['dm'][:]
    distance_map[distance_map>0] = 255
    distance_map = cv2.cvtColor(distance_map, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output, distance_map)

if __name__ == "__main__":
    filepath = "g1_0157.h5"

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(cur_dir, '..', 'datasets', 'ofda')
    src_dir = os.path.join(base_path, '..', '..')
    dst_dir = os.path.join(base_path, 'train', 'segments')
    
    for filename in glob.glob(os.path.join(src_dir,"*.h5")):
        basename = os.path.basename(filename)
        target = os.path.join(dst_dir, basename.replace(".h5",".png"))
        dm2seg(filename, target)

    # with h5py.File(filepath, "r") as hf:
    #     distance_map = hf['dm'][:]

    # distance_map[distance_map>0] = 255

    # print(type(distance_map))
    # print(np.unique(distance_map))

    # distance_map = cv2.cvtColor(distance_map, cv2.COLOR_GRAY2RGB)

    # print(distance_map.shape)

    # # cv2.imshow("distance map", distance_map)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # cv2.imwrite("output.png", distance_map)
# make synthetic dataset
import os
from fiberrandom import FiberSample

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
            
def groups(n, m):
    x = n//m
    y = n%m

    res = []
    for i in range(1,m+1):
        if i==m and y>0:
            x += y
        res += [i]*x

    return res


if __name__ == '__main__':
    # test_img_path = "/Users/alain/Documents/desarrollo/dmnet/datasets/synthetic/test/images"
    # test_gt_path = "/Users/alain/Documents/desarrollo/dmnet/datasets/synthetic/test/masks"

    # initDirectory(test_img_path)
    # initDirectory(test_gt_path)

    # sample = FiberSample(256,256,printout=False)
    # total = 10
    # for i in range(total):
    #     sample.setFibers((3,15))
    #     sample.setDiameters((3,9))
    #     sample.createDistanceMapSample()
    #     sample.saveDistanceMapSample(test_img_path, test_gt_path, i, masks_h5=True)

    """
    img_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/test/images"
    gt_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/test/masks"
    
    initDirectory(img_path)
    initDirectory(gt_path)

    sample = FiberSample(256,256,printout=False)
    total = 200
    for i in range(total):
        sample.setFibers((3,15))
        sample.setDiameters((3,9))
        sample.createDistanceMapSample()
        sample.saveDistanceMapSample(img_path, gt_path, i, masks_h5=True)
    """
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(cur_dir, '../datasets/simulated/images')
    gt_path = os.path.join(cur_dir, '../datasets/simulated/masks')
    
    initDirectory(img_path)
    initDirectory(gt_path)

    #sample = FiberSample(189,198,printout=False)
    sample = FiberSample(189,189,printout=False)
    #total = 2000
    total = 2000
    
    fiber_ranges = groups(total,15)
    
    for i in range(total):
        #sample.setFibers((1,15))
        sample.setFibers((fiber_ranges[i],fiber_ranges[i]))
        sample.setDiameters((3,9))
        #sample.createSampleAndDistanceMap()
        #sample.saveDistanceMapSample(img_path, gt_path, i, masks_h5=True)
        sample.createSampleAndSkeleton()
        sample.saveSampleAndSkeleton(img_path, gt_path, i)
    
    
    
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
    """
    genera una lista de n valores entre 1 y m proporcionalmente
    """
    x = n//m
    y = n%m

    res = []
    for i in range(1,m+1):
        if i==m and y>0:
            x += y
        res += [i]*x

    return res


if __name__ == '__main__':
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(cur_dir, '../datasets/simulated/train/images')
    gt_path = os.path.join(cur_dir, '../datasets/simulated/train/masks')
    
    initDirectory(img_path)
    initDirectory(gt_path)

    sample = FiberSample(189,189,printout=False)
    total = 43 
    
    fiber_ranges = groups(total,15)
    
    for i in range(total):
        sample.setFibers((fiber_ranges[i],fiber_ranges[i]))
        sample.setDiameters((3,9))
        sample.createSampleAndSkeleton()
        sample.saveSampleAndSkeleton(img_path, gt_path, i)

    # test images
    img_path = os.path.join(cur_dir, '../datasets/simulated/test/images')
    gt_path = os.path.join(cur_dir, '../datasets/simulated/test/masks')
    
    initDirectory(img_path)
    initDirectory(gt_path)

    sample = FiberSample(189,189,printout=False)
    total = 11
    
    fiber_ranges = groups(total,11)
    
    for i in range(total):
        sample.setFibers((fiber_ranges[i],fiber_ranges[i]))
        sample.setDiameters((3,9))
        sample.createSampleAndSkeleton()
        sample.saveSampleAndSkeleton(img_path, gt_path, i)
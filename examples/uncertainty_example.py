from glob import glob
from keras.models import load_model
import sys
sys.path.append('..')
from BioExp.uncertainty import uncertainty
from BioExp.helpers.utils import load_vol_brats
from BioExp.helpers.losses import *

# path_HGG = glob('/home/pi/Projects/beyondsegmentation/HGG/**')
# path_LGG = glob('/home/pi/Projects/beyondsegmentation/LGG**')

test_path= '/home/parth/Interpretable_ML/BraTS_2018/val/**'
np.random.seed(2022)
# np.random.shuffle(test_path)
model = load_model('/home/parth/Interpretable_ML/saved_models/densedrop/densedrop.h5', 
                custom_objects={'gen_dice_loss':gen_dice_loss,
                                'dice_whole_metric':dice_whole_metric,
                                'dice_core_metric':dice_core_metric,
                                'dice_en_metric':dice_en_metric})

model.load_weights('/home/parth/Interpretable_ML/saved_models/densedrop/model_lrsch.hdf5', by_name = True)

model_no_drop = load_model('/home/parth/Interpretable_ML/saved_models/densenet/densenet121.h5', 
                custom_objects={'gen_dice_loss':gen_dice_loss,
                                'dice_whole_metric':dice_whole_metric,
                                'dice_core_metric':dice_core_metric,
                                'dice_en_metric':dice_en_metric})

model_no_drop.load_weights('/home/parth/Interpretable_ML/saved_models/densenet/densenet.55_0.522.hdf5', by_name = True)

if __name__ == '__main__':
    list_ = []
    for volume in [10]:
        for slice_ in range(78,79):
            test_image, gt = load_vol_brats(glob(test_path)[volume], slice_, pad = 8)

            D = uncertainty(test_image)
            
            # for aleatoric
            mean, var = D.aleatoric(model_no_drop, iterations = 50)

            D.save(mean, var, gt)
           
            # for epistemic
            mean, var = D.epistemic(model, iterations = 50)

            D.save(mean, var, gt)
 
            # # for combined
            mean, var = D.combined(model, iterations = 50)
            
            D.save(mean, var, gt)

            print (np.mean(var, axis=(0,1,2)))
            list_.append(np.mean(var, axis=(0,1,2)))


    list_ = np.mean(list_, axis=0)
    print (list_)

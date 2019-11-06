import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from glob import glob
import sys
import os
sys.path.append('../../')
from BioExp.helpers import utils
from BioExp.spatial import ablation
#from BioExp.helpers.losses import *
from BioExp.spatial.losses import *
from BioExp.helpers.metrics import *
import pickle
from lucid.modelzoo.vision_base import Model
from BioExp.concept.feature import Feature_Visualizer
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
from BioExp.helpers.pb_file_generation import generate_pb
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

seq = 'flair'


data_root_path = '../../sample_vol/'

seq_to_consider = ['flair']#, 't1c', 't2', 't1']

layers_to_consider = ['conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9', 'conv2d_10', 'conv2d_11', 'conv2d_12', 'conv2d_13', 'conv2d_14', 'conv2d_15', 'conv2d_16', 'conv2d_17', 'conv2d_18', 'conv2d_19', 'conv2d_20', 'conv2d_21']
input_name = 'input_1'

infoclasses = {}
for i in range(1): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3)

for seq in seq_to_consider:

	model_pb_path = '../../../saved_models/model_{}/model.pb'.format(seq)	
	model_path = '../../../saved_models/model_{}/model-archi.h5'.format(seq)
	weights_path = '../../../saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)


	model = load_model(model_path, 
			custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,
			'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})

	print("Models Loaded")
	for layer_name in layers_to_consider:
		n_classes = len(infoclasses)
		metric = dice_label_coef


		if 'conv2d' in layer_name:	
			print(layer_name)
			for file in tqdm(glob(data_root_path +'*')[:10]):

				test_image, gt = utils.load_vol_brats(file, slicen=78)
				test_image = test_image[:, :, 0].reshape((1, 240, 240, 1))	

				A = ablation.Ablation(model, weights_path, metric, layer, test_image, gt, classes = infoclasses)
				ablation_dict = A.ablate_filter(32)

				try:
					values = pd.concat([values, pd.DataFrame(ablation_dict['value'])], axis=1)	
				except:
					values = pd.DataFrame(ablation_dict['value'], columns = ['value'])


			mean_value = values.mean(axis=1)
			for key in ablation_dict.keys():
				if key != 'value':
					try:
						layer_df = pd.concat([layer_df, pd.DataFrame(ablation_dict[key], columns = [key])], axis=1)	
					except:
						layer_df = pd.DataFrame(ablation_dict[key], columns = [key])

			layer_df = pd.concat([layer_df, mean_value.rename('value')], axis=1)	
			sorted_df = layer_df.sort_values(['class_list', 'value'], ascending=[True, False])

			for i in range(n_classes):
				save_path = 'Ablation/unet_{}/'.format(seq) + layer_name
				os.makedirs(save_path, exist_ok=True)

				for class_ in infoclasses.keys()
					class_df = sorted_df
					class_df.to_csv(save_path +'/{}.csv'.format(class_))

					if not os.path.exists(model_pb_path):
					    print (model.summary())
					    layer_name = 'conv2d_21'# str(input("Layer Name: "))
					    generate_pb(model_path, layer_name, model_pb_path, weights_path)

					input_name = 'input_1' #str(input("Input Name: "))
					class Load_Model(Model):
					    model_path = model_pb_path
					    image_shape = [None, 1, 240, 240]
					    image_value_range = (0, 1)
					    input_name = input_name


					graph_def = tf.GraphDef()
					with open(model_pb_path, "rb") as f:
					    graph_def.ParseFromString(f.read())

					print ("==========================")
					texture_maps = []

					# pdb.set_trace()
					counter  = 0
					save_pth = os.path.join('lucid/unet_{}/{}'.format(seq, class_))
					os.makedirs(save_pth, exist_ok=True)

					regularizer_params = {'L1':1e-5, 'rotate':10}

					E = Feature_Visualizer(Load_Model, 
								savepath = save_pth, 
								regularizer_params = regularizer_params)
					
					nidx = 5
					feature_maps = class_df['filter'].values[:nidx]
					layers = [model.layers[layer].name]*nidx				
					for layer_, feature_ in zip(layers, feature_maps):


					    print (layer_, feature_)
					    # Initialize a Visualizer Instance
					    texture_maps.append(E.run(layer = layer_, # + '_' + str(feature_), 
									channel = feature_, 
									class_ = class_,
									transforms = True)) 
					    counter += 1


					json = {'textures': texture_maps, 
						'class_info': 'whole', 
						'features': feature_maps, 
						'layer_info': layers}

					import pickle
					pickle_path = os.path.join('lucid/unet_{}/'.format(seq))
					os.makedirs(pickle_path, exist_ok=True)
					file_ = open(os.path.join(pickle_path, 'all_info'), 'wb')
					pickle.dump(json, file_)

			del values, layer_df, mean_value


# print(sorted_df['class_list'], sorted_df['value'])

# K.clear_session()
# # Initialize a class which loads a Lucid Model Instance with the required parameters
# from BioExp.helpers.pb_file_generation import generate_pb

# if not os.path.exists(model_pb_path):
#     print (model.summary())
#     layer_name = 'conv2d_21'# str(input("Layer Name: "))
#     generate_pb(model_path, layer_name, model_pb_path, weights_path)

# input_name = 'input_1' #str(input("Input Name: "))
# class Load_Model(Model):
#     model_path = model_pb_path
#     image_shape = [None, 1, 240, 240]
#     image_value_range = (0, 1)
#     input_name = input_name


# graph_def = tf.GraphDef()
# with open(model_pb_path, "rb") as f:
#     graph_def.ParseFromString(f.read())

# texture_maps = []

# counter  = 0
# for layer_, feature_, class_ in zip(sorted_df['layer'], sorted_df['filter'], sorted_df['class_list']):
#     # if counter == 2: break
#     K.clear_session()
    
#     # Run the Visualizer
#     print (layer_, feature_)
#     # Initialize a Visualizer Instance
#     save_pth = '/media/parth/DATA/datasets/BioExp_results/lucid/unet_{}/ablation/'.format(seq)
#     os.makedirs(save_pth, exist_ok=True)
#     E = Feature_Visualizer(Load_Model, savepath = save_pth)
#     texture_maps.append(E.run(layer = model.layers[layer].name, # + '_' + str(feature_), 
# 						 channel = feature_, transforms = True)) 
#     counter += 1


# json = {'texture':texture_maps, 'class':list(sorted_df['class_list']), 'filter':list(sorted_df['filter']), 'layer':layer, 'importance':list(sorted_df['value'])}


# pickle_path = '/media/parth/DATA/datasets/BioExp_results/lucid/unet_{}/ablation/'.format(seq)
# os.makedirs(pickle_path, exist_ok=True)
# file_ = open(os.path.join(pickle_path, 'all_info'), 'wb')
# pickle.dump(json, file_)

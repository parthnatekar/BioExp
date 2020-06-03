import keras
import random
import numpy as np
from glob import glob
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import keras.backend as K
import tensorflow as tf
from keras.utils import to_categorical
from tqdm import tqdm
import sys
sys.path.append('..')
from helpers.losses import *
from helpers.utils import load_vol_brats

class intervention():

	def __init__(self, model, test_path):
		self.model = model
		self.vol_path = glob(test_path)
		self.test_image, self.gt = load_vol_brats(self.vol_path[3], slicen = 78, pad = 0)

	def mean_swap(self, plot = True, save_path='/home/parth/Interpretable_ML/BioExp/results/RCT'):

		channel = 3

		f_index = 0

		test_image, gt = load_vol_brats(self.vol_path[f_index], slicen = 78, pad = 0)

		prediction = np.argmax(self.model.predict(test_image[None, ...]), axis = -1)[0]
		n_classes = (len(np.unique(prediction)))
		corr = np.zeros((n_classes, n_classes))
		slices = [78]

		plt.figure(figsize = (20,20))

		for vol in range(len(test_path)):
			for slicen in slices:

				test_image, gt = load_vol_brats(self.vol_path[vol], slicen = slicen, pad = 0)

				prediction = np.argmax(self.model.predict(test_image[None, ...]), axis = -1)[0]
				print("Original Dice Whole:", dice_whole_coef(prediction, gt))

				class_dict = {0:'bg', 1:'core', 2:'edema', 3:'enhancing'}

				corr_temp = np.zeros((n_classes, n_classes))
				for i in range(n_classes):
					for j in range(n_classes):
						new_mean = np.mean(test_image[gt == i], axis = 0)
						old_mean = np.mean(test_image[gt == j], axis = 0)
						test_image_intervention = np.copy(test_image)
						test_image_intervention[gt == j] += (new_mean - old_mean)
						prediction_intervention = np.argmax(self.model.predict(test_image_intervention[None, ...]), axis = -1)[0]

						corr[i,j] += dice_label_coef(prediction, gt, (j,)) - dice_label_coef(prediction_intervention, gt, (j,))
						corr_temp[i,j] += dice_label_coef(prediction, gt, (j,)) - dice_label_coef(prediction_intervention, gt, (j,))
						
						if plot == True:
							plt.subplot(n_classes, n_classes, 1+4*i+j)
							plt.xticks([])
							plt.yticks([])
							plt.title("{} --> {}, Dice Change={}".format(class_dict[j], class_dict[i], "{0:.2f}".format(-corr[i,j])))
							plt.imshow(prediction_intervention, cmap = plt.cm.RdBu, vmin = 0, vmax = 3)
							plt.colorbar()
				print(corr_temp)#/(vol*len(slices))

		np.set_printoptions(precision = 2)
		plt.rcParams.update({'font.size': 24})

		intervention_importance = corr /(len(self.vol_path)*len(slices))
		print(intervention_importance)
		os.makedirs(save_path, exist_ok = True)
		# np.save(save_path + '/mean_swap_all_images.npy', intervention_importance)
		if plot == True:
			plt.show()

	def blocks(self):

		test_image, gt = load_vol_brats(self.vol_path[1], slicen = 78, pad = 8)

		prediction = np.argmax(self.model.predict(test_image[None, ...]), axis = -1)[0]
		n_classes = (len(np.unique(prediction)))
		corr = np.zeros((n_classes, n_classes))
		slices = [78]

		intervention_image = np.empty(test_image.shape)

		for _modality in range(4):
			for i in range(2):
				for j in range(2):
					try:
						intervention_image[:,:,_modality][test_image.shape[0]//2*i:test_image.shape[0]//2*(i+1), 
						test_image.shape[1]//2*j:test_image.shape[1]//2*(j+1)].fill(np.mean(test_image[gt == 2*i+j], axis = 0)[_modality])
					except Exception as e:
						print(e)

		prediction_intervention = model.predict(intervention_image[None, ...])
		plt.imshow(intervention_image[:, :, 0])
		plt.colorbar()
		plt.show()
		plt.imshow(np.argmax(prediction_intervention, axis = -1)[0], vmin=0, vmax=3)
		plt.colorbar()
		plt.show()

	def adverserial(self, epochs=100, epsilon = 0.01, mode = 'gradient', plot=False, test_image=None, gt=None):

		sess = K.get_session()

		keras.layers.core.K.set_learning_phase(0)

		image = test_image[None, ...] # if test_image is not None else self.test_image[None, ...]
		gt = gt[None, ...] # if gt is not None else self.gt[None, ...]

		noise = np.zeros_like(image)

		adverserial_image = image.copy()

		if mode == 'gradient':
			loss = keras.losses.categorical_crossentropy(self.model.output, tf.convert_to_tensor(to_categorical(gt, num_classes=4)))
		elif mode == 'random':
			loss = -keras.losses.categorical_crossentropy(self.model.output, 
				tf.convert_to_tensor(self.generate_random_classification(mode='random')))
		elif mode =='swap':
			loss = -keras.losses.categorical_crossentropy(self.model.output, 
				tf.convert_to_tensor(self.generate_random_classification(mode='swap')))

		grads = K.gradients(loss, self.model.input)

		delta = K.sign(grads[0])

		noise = noise + delta

		adverserial_image = adverserial_image+epsilon*delta

		adverserial_image, noise_ar, delta_ = sess.run([adverserial_image, noise, delta], feed_dict={self.model.input: image})

		delta_image_perc = (np.mean(np.abs(image - adverserial_image))*100)/np.ptp(image)

		delta_dice_perc = (dice_whole_coef(self.model.predict(image).argmax(axis=-1), 
			gt) - dice_whole_coef(self.model.predict(adverserial_image).argmax(axis=-1), 
			gt))*100/dice_whole_coef(self.model.predict(image).argmax(axis=-1), 
			gt)

		# print("perc. change in image:{}, perc. change in dice:{}, Sensitivity:{}".format(delta_image_perc,
		# delta_dice_perc, delta_dice_perc/delta_image_perc))

		imshape = image.shape[1]

		if plot==True:

			plt.figure(figsize = (40,10))

			plt.rcParams.update({'font.size': 34})
			plt.subplot(1,4,1)
			plt.title("Original image")
			plt.imshow(image[:, :, :, 0].reshape((imshape, imshape))) 
			plt.xticks([])
			plt.yticks([])       
			# plt.subplot(1,6,2)
			# plt.title("Added Noise")
			# plt.imshow(noise_ar[:, :, :, 0].reshape((imshape, imshape)))
			# plt.xticks([])
			# plt.yticks([])
			plt.subplot(1,4,2)
			plt.title("Image + Noise, % Change = {}".format("{0:.2f}".format(delta_image_perc)))	
			plt.imshow(adverserial_image[:, :, :, 0].reshape((imshape, imshape)))
			plt.xticks([])
			plt.yticks([])
			# plt.subplot(1,6,4)
			# plt.title("Ground Truth")
			# plt.imshow(self.gt, vmin = 0, vmax=3)
			# plt.xticks([])
			# plt.yticks([])
			plt.subplot(1,4,3)
			plt.title("Old Seg, Dice = {}".format("{0:.2f}".format(dice_whole_coef(self.model.predict(image).argmax(axis=-1), gt))))
			plt.imshow(np.argmax(self.model.predict(image), axis = -1).reshape((imshape, imshape)), vmin = 0, vmax=3)
			plt.xticks([])
			plt.yticks([])
			plt.subplot(1,4,4)
			plt.title("New Seg, Dice={}, Sensitivity={}".format("{0:.2f}".format(dice_whole_coef(self.model.predict(adverserial_image).argmax(axis=-1), 
				gt)), "{0:.2f}".format(delta_dice_perc/delta_image_perc)))
			plt.imshow(np.argmax(self.model.predict(adverserial_image), axis = -1).reshape((imshape, imshape)), vmin = 0, vmax=3)
			plt.xticks([])
			plt.yticks([])
			plt.tight_layout(pad=0)
			plt.show()
			# plt.savefig('/home/parth/Interpretable_ML/Adverserial Examples/adv_{}.png'.format(epsilon))

		return(delta_image_perc, delta_dice_perc, delta_dice_perc/delta_image_perc)

	def generate_random_classification(self, mode='random'):

		if mode == 'random':

			true_target = self.gt.flatten()

			true_target[true_target==4] = 3

			index_list = [0, 1, 2, 3]

			adverserial_random = np.zeros_like(true_target)

			for i in range(adverserial_random.shape[0]):

				adverserial_random[i] = np.random.choice(np.setdiff1d(index_list, true_target[i]))
			    
			print("Target image")
			plt.imshow(adverserial_random.reshape((256, 256)), vmin=0., vmax=3.)
			plt.show()

			return to_categorical(adverserial_random, num_classes=4).reshape(self.test_image.shape)

		elif mode == 'swap':

			true_target = self.gt.flatten()

			true_target[true_target==4] = 3

			index_list = [0, 1, 2, 3]

			adverserial_random = np.zeros_like(true_target)

			for i in index_list:

				adverserial_random[true_target == i] = np.random.choice(np.setdiff1d(index_list, i))

			print("Target image")
			plt.imshow(adverserial_random.reshape((256, 256)), vmin=0., vmax=3.)
			plt.show()

			return to_categorical(adverserial_random, num_classes=4).reshape(self.test_image.shape)


if __name__ == "__main__":

	model = load_model('/home/parth/Interpretable_ML/saved_models/SimUnet/model_lrsch.hdf5', 
                custom_objects={'gen_dice_loss':gen_dice_loss,
                                'dice_whole_metric':dice_whole_metric,
                                'dice_core_metric':dice_core_metric,
                                'dice_en_metric':dice_en_metric})

	model.load_weights('/home/parth/Interpretable_ML/saved_models/SimUnet/SimUnet.40_0.060.hdf5')

	I = intervention(model, '/media/parth/DATA/datasets/brats_2018/val/**')

	test_path = glob('/media/parth/DATA/datasets/brats_2018/val/**')

	average_change = []

	for epsilon in [0.7]: #, 0.07, 0.21, 0.7]:
		for i in tqdm(range(len(test_path))):

			test_image, gt = load_vol_brats(test_path[i], slicen = 78, pad = 0)
			if len(np.unique(gt)) == 4:
				print(len(np.unique(gt)))
				# I.blocks('/home/parth/Interpretable_ML/BioExp/sample_vol/brats/**')
				adv = I.adverserial(epsilon = epsilon, mode='gradient', test_image=test_image, gt=gt)
				if adv[1] > 0:
					average_change.append(adv)
					print(adv)

		print(np.mean(average_change, axis = 0))

	# I.generate_random_classification(mode='swap')
	# I.mean_swap(plot = False)


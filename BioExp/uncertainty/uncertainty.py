import keras
import random
import numpy as np
from glob import glob
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model

import os
import imgaug as ia
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import matplotlib.gridspec as gridspec

import sys
sys.path.append('..')
# from helpers.losses import *
# from helpers.utils import load_vol_brats
# from evaluation_metrics import *


class uncertainty():
    """
    estimates model and data uncertanity

    """
    
    def __init__(self, test_image, savepath=None):
        """
        test_image: image for uncertanity estimation
        savepath  : path to save uncertanity images

	"""
        self.test_image = test_image
        self.savepath   = savepath

    def save(self, mean, var, gt):
        """
        mean: mean image
        var : variance image
        """
        # print(self.test_image.shape)
        plt.figure(figsize=(10, 40))
        gs = gridspec.GridSpec(1, 4)
        gs.update(wspace=0.02, hspace=0.02)

        ax = plt.subplot(gs[0, 0])
        im = ax.imshow(self.test_image[:, :, 2].reshape((256, 256)), cmap = 'Greys_r')
        ax = plt.subplot(gs[0, 1])
        im = ax.imshow(gt.reshape((256, 256)), vmin=0., vmax=3.)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        ax = plt.subplot(gs[0, 2])
        im = ax.imshow(np.argmax(mean, axis = -1).reshape((256, 256)), vmin=0., vmax=3.)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        ax = plt.subplot(gs[0, 3])
        im = ax.imshow(var[:, :, :, 2].reshape((256, 256)), cmap=plt.cm.RdBu_r)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )

        if self.savepath:
            plt.savefig(self.savepath, bbox_inches='tight')
        else:
            plt.show()


    def aleatoric(self, model, iterations = 1000):
        """
	    estimates data uncertanity
        iterations: montecarlo sample iterations

        """
        self.aug = iaa.Sequential([
                        iaa.Affine(
                        rotate=(-1,1),
                        # translate_px={"x": (-1, 1), "y": (-1, 1)}
                        ),
                        # iaa.SaltAndPepper(0.001),
                        iaa.AdditiveGaussianNoise(scale=0.2),
                        iaa.Noop(),
                        iaa.MotionBlur(k=3, angle = [-1, 1])
                    ], random_order=False)

        predictions = []
        
        for i in range(iterations):
            aug_image = self.aug.augment_images(self.test_image.astype(np.float32))
            predictions.append(model.predict(aug_image[None, ...]))
            
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis = 0)
        var = np.var(predictions, axis = 0)
        
        return mean, var


    def epistemic(self, model, iterations=1000, dropout=0.5):
        """
        estimates model uncertanity
        iterations: montecarlo sample iterations

        """
        predictions = []
        
        for i in range(iterations):
            predictions.append(model.predict(self.test_image[None, ...]))
            
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis = 0)
        var = np.var(predictions, axis = 0)
        if np.sum(var) == 0: raise ValueError("Model trained without dropouts")
        

        return mean, var

    def combined(self, model, iterations=1000, dropout=0.5):
        """
	    estimates combined uncertanity
        iterations: montecarlo sample iterations

        """
        return self.aleatoric(model, iterations=iterations)




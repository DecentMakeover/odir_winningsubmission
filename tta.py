from tqdm import tqdm
from math import ceil
from keras.models import model_from_json
import numpy as np
from keras.activations import elu
import cv2
import time
import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.activations import elu
from keras.optimizers import Adam
from keras.models import Sequential
from keras.engine import Layer, InputSpec
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score
from keras.models import model_from_json
import efficientnet.keras as efn

class GroupNormalization(Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and 
    returns the a preprocessed image with 
    3 channels
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image
    
exp_name ='exp_3'
IMG_WIDTH = 456
IMG_HEIGHT= 456
SEED = 123456
test_images_path  = 'extra_data/odir/ODIR-5K_Testing_Images/'
json_file = 'saved_models/exp_3/exp_7model_ef5fn_on_dr.json'
weights_path="saved_models/exp_3/exp_7wieghts_ef5dr_fn.h5"
sample_subm_path  = 'extra_data/odir/XYZ_ODIR.csv'
json_file = open(json_file, 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(weights_path)
print("Loaded model from disk")

df_sample = pd.read_csv(sample_subm_path)
df_test_left = pd.DataFrame() 
df_test_left['id']     = df_sample.ID
df_test_left['pic_id'] = df_sample.ID.apply(lambda x: str(x)+"_left.jpg")
df_test_left.to_csv('extra_data/odir/test_df_left.csv')

df_test_right = pd.DataFrame() 
df_test_right['id']     = df_sample.ID
df_test_right['pic_id'] = df_sample.ID.apply(lambda x: str(x)+"_right.jpg")
df_test_right.to_csv('extra_data/odir/train_df_right.csv')

print('DF were created.')

BATCH_SIZE =3
IMG_WIDTH  = 456
IMG_HEIGHT = 456
# Add Image augmentation to our generator
train_datagen = ImageDataGenerator(rotation_range=360,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split=0.15,
                                   preprocessing_function=preprocess_image, 
                                   rescale=1 / 128.)

### -------- left -------------------------------------
left_test_generator=train_datagen.flow_from_dataframe(dataframe=df_test_left, 
                                                directory = test_images_path,
                                                x_col="pic_id",
                                                target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                batch_size=1,
                                                shuffle=False, 
                                                class_mode=None, seed=SEED) 
### -------- rigth ------------------------------------
right_test_generator=train_datagen.flow_from_dataframe(dataframe=df_test_right, 
                                                directory = test_images_path,
                                                x_col="pic_id",
                                                target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                batch_size=1,
                                                shuffle=False, 
                                                class_mode=None, seed=SEED)
# -----------------------------------------------------  

from imgaug import augmenters as iaa
sometimes_more = lambda aug: iaa.Sometimes(0.7, aug)
sometimes = lambda aug: iaa.Sometimes(0.3, aug)
sometimes_less = lambda aug: iaa.Sometimes(0.2, aug)

seq = iaa.Sequential([
    sometimes(
        iaa.OneOf([
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.9, 1.1), per_channel=0.5),
            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5),
            
        ])
            ),
    sometimes_more(
        iaa.OneOf([
            iaa.CropToFixedSize(1024, 1024),
            iaa.CropToFixedSize(1536, 1536),
            iaa.Affine(scale={"x": (2, 1.2), "y": (0.8, 1.2)},
                       rotate=(-10, 10))
        ])
            ),
    sometimes_less(
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AverageBlur(k=(2, 7)), 
            # iaa.MedianBlur(k=(5, 11)),
        ])
                ),
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Flipud(0.5)] ,random_order=True)

# import pdb;pdb.set_trace()
for image_count, image in enumerate(right_test_generator):
    print(image_count)

f_pred_right = np.empty((len(right_test_generator), 8))
for image_count, image in enumerate(right_test_generator):
    image_batch = np.empty((3,456, 456, 3))
    for index in range(3):
        # images_ = seq(images=[np.squeeze(image,0)])
        # image_batch[index,: ] = images_[0]
        image_batch[index,: ] = image

    predict_batch = model.predict(image_batch)
    f_pred = np.mean(predict_batch)
    # f_pred_right[image_count, :] = f_pred
    if image_count ==499:
        import pdb;pdb.set_trace()
        break 


print('Done Right')
f_pred_left = np.empty((len(left_test_generator), 8))
for image_count, image in enumerate(left_test_generator):
    image_batch = np.empty((3,456, 456, 3))
    for index in range(3):
        # images_ = seq(images=[np.squeeze(image,0)])
        # image_batch[index,: ] = images_[0]
        image_batch[index,: ] = image
    predict_batch = model.predict(image_batch)
    f_pred = np.mean(predict_batch)
    f_pred_left[image_count, :] = f_pred
    if image_count ==499:
        break

final_predict = 0.5*f_pred_left+0.5*f_pred_right

df_submit = pd.DataFrame()
df_submit['ID'] = df_test_right.id
df_submit['N'] = final_predict[:,0]
df_submit['D'] = final_predict[:,1]
df_submit['G'] = final_predict[:,2]
df_submit['C'] = final_predict[:,3]
df_submit['A'] = final_predict[:,4]
df_submit['H'] = final_predict[:,5]
df_submit['M'] = final_predict[:,6]
df_submit['O'] = final_predict[:,7]
df_submit.to_csv('Andy_ODIRdr_{}.csv'.format(exp_name),index=False)
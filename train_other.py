#########################
bubbles = False
if bubbles:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=config))
#########################


#imports
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ## files paths and models wieghts

def set_paths(dataset, exp):
    csv_path = {'odir':'ODIR-5K_Training_Annotations(Updated)_V2.xlsx', 'db': 'extra_data/bd/train.csv', 'dr_train': 'extra_data/dr/train.csv', 'dr_test': 'extra_data/dr/test.csv'}
    train_images_path = 'extra_data/{}/train/'.format(dataset)
    test_images_path  = 'extra_data/odir/ODIR-5K_Testing_Images/'
    df_path           = csv_path[dataset]
    sample_subm_path  = 'extra_data/odir/XYZ_ODIR.csv'
    SAVED_MODEL_NAME  = './saved_models/{}_wieghts.h5'.format(exp)
    json_path = "./saved_models/{}_model.json".format(exp)
    return train_images_path, test_images_path ,df_path, sample_subm_path ,SAVED_MODEL_NAME, json_path

dataset_name = 'odir'
exp_name = 'exp_15'
train_images_path, test_images_path ,df_path, sample_subm_path ,SAVED_MODEL_NAME, json_path = set_paths(dataset_name, exp_name)
if dataset_name == 'dr_test':
    train_images_path = 'extra_data/dr/test/'
elif dataset_name =='dr_train':
    train_images_path = 'extra_data/dr/train/'
import os
import sys
import efficientnet.keras as efn 
SEED = 123456
import os
import random as rn
import numpy as np
from tensorflow import set_random_seed

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
set_random_seed(SEED)
rn.seed(SEED)

if dataset_name != 'odir':
    train_df = pd.read_csv(df_path)
else:
    train_df = pd.read_excel(df_path)
    Labels_list=['N','D','G','C','A','H','M','O']
    train_df.head(5)
    train_df.describe()
    df1 = pd.DataFrame()
    df1['pic_id'] = train_df['Left-Fundus']

    df1['N'] = np.where(train_df['Left-Diagnostic Keywords']=='normal fundus', 1, train_df.N)
    df1['D'] = np.where(train_df['Left-Diagnostic Keywords']=='normal fundus', 0, train_df.D)
    df1['G'] = np.where(train_df['Left-Diagnostic Keywords']=='normal fundus', 0, train_df.G)
    df1['C'] = np.where(train_df['Left-Diagnostic Keywords']=='normal fundus', 0, train_df.C)
    df1['A'] = np.where(train_df['Left-Diagnostic Keywords']=='normal fundus', 0, train_df.A)
    df1['H'] = np.where(train_df['Left-Diagnostic Keywords']=='normal fundus', 0, train_df.H)
    df1['M'] = np.where(train_df['Left-Diagnostic Keywords']=='normal fundus', 0, train_df.M)
    df1['O'] = np.where(train_df['Left-Diagnostic Keywords']=='normal fundus', 0, train_df.O)


    df2 = pd.DataFrame()
    df2['pic_id'] = train_df['Right-Fundus']
    df2['Right-Diagnostic Keywords'] = train_df['Right-Diagnostic Keywords']
    df2['N'] = np.where(train_df['Right-Diagnostic Keywords']=='normal fundus', 1, train_df.N)
    df2['D'] = np.where(train_df['Right-Diagnostic Keywords']=='normal fundus', 0, train_df.D)
    df2['G'] = np.where(train_df['Right-Diagnostic Keywords']=='normal fundus', 0, train_df.G)
    df2['C'] = np.where(train_df['Right-Diagnostic Keywords']=='normal fundus', 0, train_df.C)
    df2['A'] = np.where(train_df['Right-Diagnostic Keywords']=='normal fundus', 0, train_df.A)
    df2['H'] = np.where(train_df['Right-Diagnostic Keywords']=='normal fundus', 0, train_df.H)
    df2['M'] = np.where(train_df['Right-Diagnostic Keywords']=='normal fundus', 0, train_df.M)
    df2['O'] = np.where(train_df['Right-Diagnostic Keywords']=='normal fundus', 0, train_df.O)

    # train_df = pd.read_excel(df_path)
    # Labels_list=['N','D','G','C','A','H','M','O']
    # train_df.head(5)
    # train_df.describe()
    # df1 = pd.DataFrame()
    # df1['pic_id'] = train_df['Left-Fundus']
    # df1['N'] = train_df.N
    # df1['D'] = train_df.D
    # df1['G'] = train_df.G
    # df1['C'] = train_df.C
    # df1['A'] = train_df.A
    # df1['H'] = train_df.H
    # df1['M'] = train_df.M
    # df1['O'] = train_df.O


    # df2 = pd.DataFrame()
    # df2['pic_id'] = train_df['Right-Fundus']
    # df2['N'] = train_df.N
    # df2['D'] = train_df.D
    # df2['G'] = train_df.G
    # df2['C'] = train_df.C
    # df2['A'] = train_df.A
    # df2['H'] = train_df.H
    # df2['M'] = train_df.M
    # df2['O'] = train_df.O

    df = pd.concat([df1,df2])
    df.to_csv('extra_data/odir/train_df.csv')
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size = 0.1, stratify=df[['N','D','G','C','A','H','M','O']], random_state = 73)

# ------------------------------------------
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

def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator
    """
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
class Metrics(Callback):
    """
    A custom Keras callback for saving the best model
    according to the Quadratic Weighted Kappa (QWK) metric
    """
    def on_train_begin(self, logs={}):
        """
        Initialize list of QWK scores on validation data
        """
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Gets QWK score on the validation data
        """
        # Get predictions and convert to integers
        y_pred, labels = get_preds_and_labels(model, val_generator)
        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
        # We can use sklearns implementation of QWK straight out of the box
        # as long as we specify weights as 'quadratic'
        _val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
        self.val_kappas.append(_val_kappa)
        print(f"val_kappa: {round(_val_kappa, 4)}")
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save(SAVED_MODEL_NAME)
        return
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
class RAdam(keras.optimizers.Optimizer):
    """RAdam optimizer.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay for each param.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.
        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.
        min_lr: float >= 0. Minimum learning rate after warmup.
    # References
        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,
                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = self.total_steps - warmup_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr * (1.0 - K.minimum(t, decay_steps) / decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t) + self.epsilon)

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t > 5, r_t * m_corr_t / v_corr_t, m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
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
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------

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
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
from imgaug import augmenters as iaa
sometimes_more = lambda aug: iaa.Sometimes(0.7, aug)
sometimes = lambda aug: iaa.Sometimes(0.3, aug)
sometimes_less = lambda aug: iaa.Sometimes(0.2, aug)
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# seq = iaa.Sequential([
#    sometimes(
#        iaa.OneOf([
#            iaa.Add((-10, 10), per_channel=0.5),
#            iaa.Multiply((0.9, 1.1), per_channel=0.5),
#            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5),
           
#        ])
#            ),
#    sometimes_more(
#        iaa.OneOf([
#            iaa.CropToFixedSize(1024, 1024),
#            iaa.CropToFixedSize(1536, 1536),
#            iaa.Affine(scale={"x": (2, 1.2), "y": (0.8, 1.2)},
#                       rotate=(-10, 10))
#        ])
#            ),
#    sometimes_less(
#        iaa.OneOf([
#            iaa.GaussianBlur(sigma=(0.0, 3.0)),
#            iaa.AverageBlur(k=(2, 7)), 
#            # iaa.MedianBlur(k=(3, 11)),
#        ])
#                ),
#    iaa.Fliplr(0.5),
#    iaa.Crop(percent=(0, 0.1)),
#    iaa.Flipud(0.5)] ,random_order=True)    
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# def preprocess_image(image, sigmaX=10):
#     #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #image = crop_image_from_gray(image)
#     # image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
#     images_ = seq(images=[image])
#     image   = images_[0]
#     image = cv2.resize(image, (456, 456))
#     return image

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
#-----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# custom loss function
import keras
import keras.backend as K
def custom_mse(class_weights):
    def loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        # print('y_pred:', K.int_shape(y_pred))
        # print('y_true:', K.int_shape(y_true))
        y_pred = K.reshape(y_pred, (8, 1))
        y_pred = K.dot(class_weights, y_pred)
        # calculating mean squared error
        mse = K.mean(K.square(y_pred - y_true), axis=-1)
        # print('mse:', K.int_shape(mse))
        return mse
    return loss_fixed

class_weights = K.variable([[8.77,8.66,46.0,47.0,60.0,97.0,57.0,10.0]])
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet169
effnet = DenseNet169(include_top=False, weights='imagenet')
# model = Sequential()    
# model.add(base_model)
# model.load_weights('saved_models/resnet/Resnet50_bestqwk.h5')

# effnet = efn.EfficientNetB5( weights='imagenet', include_top=False)
# Replace all Batch Normalization layers by Group Normalization layers
# for i, layer in enumerate(effnet.layers):
    # if "batch_normalization" in layer.name:
        # effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
        
# for i, layer in enumerate(effnet.layers[:-1]):
    # effnet.layers[i].train = False

model = Sequential()    
model.add(effnet)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
# model.load_weights('saved_models/resnet/Resnet50_bestqwk.h5')
model.add(Dense(8,name = 'elu', activation=elu))
model.load_weights('saved_models/exp_5/exp_5_wieghts.h5',by_name =True)
model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['mse', 'acc'])
print(model.summary())        

model_json = model.to_json()
with open(json_path, "w") as json_file:
    json_file.write(model_json)

BATCH_SIZE =7
IMG_WIDTH  = 512
IMG_HEIGHT = 512
# Add Image augmentation to our generator
train_datagen = ImageDataGenerator(rotation_range=360,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocess_image, 
                                   rescale=1 / 255.)

# Use the dataframe to define train and validation generators
if dataset_name =='odir':
    train_generator = train_datagen.flow_from_dataframe(df_train, 
                                                        x_col='pic_id', 
                                                        y_col=Labels_list,
                                                        directory = train_images_path,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='other')

    val_generator = train_datagen.flow_from_dataframe(df_test, 
                                                      x_col='pic_id', 
                                                      y_col=Labels_list,
                                                      directory = train_images_path,
                                                      target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='other')
    print('Data was loaded')

else:
    df = pd.read_csv(df_path)
    train_generator = train_datagen.flow_from_dataframe(df, 
                                                        x_col='id_code', 
                                                        y_col='diagnosis',
                                                        directory = train_images_path,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='other', 
                                                        subset='training')

    val_generator = train_datagen.flow_from_dataframe(df, 
                                                      x_col='id_code', 
                                                      y_col='diagnosis',
                                                      directory = train_images_path,
                                                      target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='other',
                                                      subset='validation')
    print('Data was loaded')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# For tracking Quadratic Weighted Kappa score
kappa_metrics = Metrics()
# Monitor MSE to avoid overfitting and save best model
filepath = "./saved_models/{val_loss:.2f}.hdf5"
checkpoints = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=15)
rlr = ReduceLROnPlateau(monitor='val_loss', 
                        factor=0.2, 
                        patience=15, 
                        verbose=1, 
                        mode='auto')
# # Begin training
model.fit_generator(train_generator,
                    # steps_per_epoch=60,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,                    
                    epochs=3000,
                    validation_data=val_generator,
                    # validation_steps = 30,
                    validation_steps = val_generator.samples // BATCH_SIZE,                    
                    callbacks=[kappa_metrics, es, rlr,checkpoints])
# Predict results for each eye
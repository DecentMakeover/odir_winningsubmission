import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import efficientnet.keras as efn 
from math import ceil
import cv2

########################
exp_name ='exp_100'
SEED = 123456
test_images_path  = 'extra_data/odir/ODIR-5K_Testing_Images/'
json_file = 'saved_models/exp_9/exp_9_model.json'
weights_path="saved_models/exp_9/exp_9_wieghts.h5"
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
########################

df_path='ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
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

df = pd.concat([df1,df2],sort=False)
from sklearn.model_selection import train_test_split
SEED=123456
train_images_path = 'extra_data/odir/train/'
IMG_WIDTH = 512
IMG_HEIGHT= 512

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
df_train, df_test = train_test_split(df, test_size = 0.1, stratify=df[['N','D','G','C','A','H','M','O']], random_state = 73)

train_datagen = ImageDataGenerator(rotation_range=360,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocess_image, 
                                   rescale=1 / 128.)
test_generator=train_datagen.flow_from_dataframe(dataframe=df_test, 
                                                directory = train_images_path,
                                                x_col="pic_id",
                                                target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                batch_size=1    ,
                                                shuffle=False, 
                                                class_mode=None, seed=SEED)

print('Starting')
preds1 = model.predict_generator(generator=test_generator,steps = ceil(df_test.shape[0]))
print('Done')
############################################
#Save csv
import os
exp_name ='exp_9'
np.savetxt(os.path.join('csv',exp_name, "exp_6_wieghts.csv"), preds1, delimiter=",")
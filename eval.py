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
json_file = 'saved_models/exp_3/exp_7model_ef5fn_on_dr.json'
weights_path="saved_models/exp_3/exp_7wieghts_ef5dr_fn.h5"
sample_subm_path  = 'extra_data/odir/XYZ_ODIR.csv'
# json_file = open(json_file, 'r')
# loaded_model_json = json_file.read()
# model = model_from_json(loaded_model_json)
# load weights into new model
# model.load_weights(weights_path)
# print("Loaded model from disk")
# ###
# json_file2 = 'saved_models/exp_5/exp_5_model.json'
# weights_path2="saved_models/exp_5/exp_5_wieghts.h5"
# json_file2 = open(json_file2, 'r')
# loaded_model_json = json_file2.read()
# model2 = model_from_json(loaded_model_json)
# # load weights into new model
# model2.load_weights(weights_path2)
# ####
# json_file3 = 'saved_models/exp_11/exp_11_model.json'
# weights_path3="saved_models/exp_11/exp_11_wieghts.h5"
# json_file3 = open(json_file3, 'r')
# loaded_model_json = json_file3.read()
# model3 = model_from_json(loaded_model_json)
# # load weights into new model
# model3.load_weights(weights_path3)
# print("Loaded model from disk")
###
# json_file4 = 'saved_models/exp_12_old/exp_11model_ef5fn_on_dr.json'
# weights_path4="saved_models/exp_12_old/exp_11wieghts_ef5dr_fn.h5"
# json_file4 = open(json_file4, 'r')
# loaded_model_json = json_file4.read()
# model4 = model_from_json(loaded_model_json)
# # load weights into new model
# model4.load_weights(weights_path4)
# print("Loaded model from disk")
########
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


# preds1 = model.predict_generator(generator=test_generator,steps = ceil(df_test.shape[0]))
print('waiting ... ')
# preds2 = model2.predict_generator(generator=test_generator,steps = ceil(df_test.shape[0]))
print('waiting ... ')
# preds3 = model3.predict_generator(generator=test_generator,steps = ceil(df_test.shape[0]))
print('No more.')
# preds4 = model4.predict_generator(generator=test_generator,steps = ceil(df_test.shape[0]))


############################################
#Save csv
import os
exp_name ='exp_100'
# np.savetxt(os.path.join('csv',exp_name, "preds1.csv"), preds1, delimiter=",")
# np.savetxt(os.path.join('csv',exp_name, "preds2.csv"), preds2, delimiter=",")                                                                                                                                             
# np.savetxt(os.path.join('csv',exp_name, "preds3.csv"), preds3, delimiter=",")                                                                                                                                             
# np.savetxt("preds4.csv", preds4, delimiter=",")   

import pandas as pd
preds1 = pd.read_csv(os.path.join('csv', exp_name, 'preds1.csv'), header= None).values
preds2 = pd.read_csv(os.path.join('csv', exp_name, 'preds2.csv'), header= None).values
preds3 = pd.read_csv(os.path.join('csv', exp_name, 'preds3.csv'), header= None).values
preds4 = pd.read_csv(os.path.join('csv', 'exp_12', '0.29.csv'), header= None).values
preds5 = pd.read_csv(os.path.join('csv', 'exp_12', 'exp_12_wieghts.csv'), header= None).values
preds6 = pd.read_csv(os.path.join('csv', 'exp_14', 'preds1.csv'), header= None).values
preds7 = pd.read_csv(os.path.join('csv', 'exp_14', '0.08.csv'), header= None).values
preds8 = pd.read_csv(os.path.join('csv', 'exp_15', 'preds3.csv'), header= None).values
preds9 = pd.read_csv(os.path.join('csv', 'exp_9', 'exp_6_wieghts.csv'), header= None).values




##########################################
# preds1 = np.where(preds1 > 0.5, 1, 0)
# preds2 = np.where(preds2 > 0.5, 1, 0)
# preds3 = np.where(preds3 > 0.5, 1, 0)
# preds3 = np.where(preds3 > 0.5, 1, 0)
# preds4 = np.where(preds4 > 0.5, 1, 0)
# preds5 = np.where(preds5 > 0.5, 1, 0)
# preds6 = np.where(preds6 > 0.5, 1, 0)
# preds7 = np.where(preds7 > 0.5, 1, 0)
# preds8 = np.where(preds8 > 0.5, 1, 0)
# preds9 = np.where(preds9 > 0.5, 1, 0)

# preds = 0.3*preds1+0.3+preds2+0.4*preds3

import itertools
data = np.arange(0.1, 1.0, 0.1)
result = list(itertools.permutations(data, 5))
result = list(np.round(result,2))
# import pdb;pdb.set_trace()
from tqdm import tqdm
for comb in tqdm(result):
    preds = comb[0]*preds8+comb[1]*preds6+comb[2]*preds3+comb[3]*preds4+comb[4]*preds9#+comb[5]*preds6+comb[6]*preds7+comb[7]*preds8
    truth = df_test[['N',  'D',  'G',  'C' , 'A',  'H',  'M'  ,'O']].values

    ##############################################################################
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import roc_auc_score
    gt = truth.flatten()
    pr = preds.flatten()
    kappa = cohen_kappa_score(gt, pr>0.5)
    f1 = f1_score(gt, pr>0.5, average='micro')
    auc = roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0#######THIS IS 5 PERCENT OFF I THINK
    ##############################################################################
    if final_score>0.87:
        print('COMB         ',comb)
        print('KAPPA SCORE  ', kappa)
        print('F1    SCORE  ', f1)
        print('AUC   SCORE  ', auc)
        print('FINAL SCORE  ', final_score)
        print()
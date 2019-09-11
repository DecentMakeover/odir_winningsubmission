import pandas as pd
import numpy as np 
exp_name='exp_9'
sample_subm_path  = 'extra_data/odir/XYZ_ODIR.csv'
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

final_predict_1 = pd.read_csv('final_predict_1.csv',header= None)
final_predict_2 = pd.read_csv('final_predict_2.csv',header= None)
final_predict_3 = pd.read_csv('final_predict_3.csv',header= None)
# import pdb;pdb.set_trace()
final_predict = 0.3*final_predict_1.values+0.3*final_predict_2.values+0.4*final_predict_3.values

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

'''
This will give 83
'''
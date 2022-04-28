#%%
import pandas as pd
# csv_path = '/home/ubuntu/Capstone/'
# df_augmented = pd.read_csv(csv_path + 'Augmented_Image_data_2.csv')
# df_augmented.head(10)
#
# #%%
# df_augmented.groupby('label').count()
#
# #%%
# # original data - entire set
# df = pd.read_csv(csv_path + 'split_audio_img.csv')
#
# #%%
# df.groupby('label').count()
#
# #%%
# # remove test set out of CapstoneZZ
# df_testimg = pd.read_csv(csv_path+"Test_Images.csv")
# df_testimg.head()
# test_img_list = df_testimg['img_name']
# df2 = df[~df['img_name'].isin(test_img_list)]
# # df2 is the original split images for cnn models (exclude testing data)
#
# #%%
# df2.groupby('label').count()
# #%%
# # Concatenate original files and augmented files
# df_augmented['Path'] = '/home/ubuntu/Capstone/Augmented_MelSpectrogram_2/' + df_augmented['Filename']
# df_augmented = df_augmented.loc[:, ~df_augmented.columns.str.contains('^Unnamed')]
# #%%
# # add a path in original dataset
# df2['Path'] = '/home/ubuntu/Capstone/split_audio_img/' + df2['img_name']
# # rename columns in df2
# df2.rename({'img_name': 'Filename'}, axis=1, inplace=True)
# #%%
# # combine original data and augmented data
# df_original_augmented = pd.concat([df2,df_augmented],ignore_index=True, sort=False )
# #%%
# # export csv of df_original_augmented
# df_original_augmented.to_csv(csv_path + 'Origin_Augmented_Training_Data_2.csv')
# #%%
# df_original_augmented.groupby(['label']).count()
# #%%
# import pandas as pd
# aug_path = '/home/ubuntu/Capstone/data_csv_0326/augmented_csv/'
# ori_path = '/home/ubuntu/Capstone/data_csv_0326/'
#
# aug_df = pd.read_csv( aug_path + 'Augmented_Images.csv')
# ori_df = pd.read_csv(ori_path + 'data_csv_0326accent_train_trim_split.csv')
# #%%
# ori_df[id] = '/home/ubuntu/Capstone/data_csv_0326/mel_spectrograms/' + ori_df['img_name']
# aug_df[id] = '/home/ubuntu/Capstone/data_csv_0326/augmented_csv/augmented_images/' + aug_df['Filename']

#%%
import argparse
def combine_data(origin_path,origin_file_name,augmented_path,augmented_csv_name,augmented_data_path, combined_csv_path):
    origin_df = pd.read_csv(origin_path + origin_file_name)
    augmented_df = pd.read_csv(augmented_path + augmented_csv_name)

    origin_df['label'] = origin_df['native_language'] # rename column
    augmented_df['path'] = augmented_data_path + '/' + augmented_df['Filename'] # add path in augmented csv file

    columns = ['path','label']
    new_origin_df = pd.DataFrame(origin_df,columns=columns) # get path and label columns only
    new_augmented_df = pd.DataFrame(augmented_df, columns=columns) # get path and label columns only

    frames = [new_origin_df,new_augmented_df]
    df_combined = pd.concat(frames) # combined two dataframes
    df_combined.to_csv(combined_csv_path + 'Origin_Augmented_Combined.csv')

    return df_combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--origin_csv_path", default=None, type=str, required=True)  # origin data csv path
    parser.add_argument("--augmented_csv_path", default=None, type=str, required=True)  # augmented data csv path
    parser.add_argument("--augmented_data_path", default=None, type=str, required=True) # this is path where you saved your augmented audio files or images
    parser.add_argument("--combined_csv_path", default=None, type=str, required=True)  # where you want to save the combined file

    parser.add_argument("--origin_csv_name", default=None, type=str, required=True)  # origin data csv file name
    parser.add_argument("--augmented_csv_name", default=None, type=str, required=True)  # augmented data csv file name

    args = parser.parse_args()
    origin_path = args.origin_csv_path
    augmented_path = args.augmented_csv_path
    combined_csv_path = args.combined_csv_path
    augmented_data_path = args.augmented_data_path

    origin_file_name = args.origin_csv_name
    augmented_csv_name = args.augmented_csv_name

    combine_data(origin_path,origin_file_name,augmented_path,augmented_csv_name,augmented_data_path, combined_csv_path)


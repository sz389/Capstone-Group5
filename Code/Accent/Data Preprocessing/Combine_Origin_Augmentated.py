#%%
import pandas as pd

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


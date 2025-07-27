import os
import pandas as pd

insta_data_dir = '/home/tsutar3/HEART/convos/SFT/empathy_10/insta/'
v1_insta_data_dir = '/home/tsutar3/HEART/convos/SFT/v1/empathy05/insta/'
twitter_data_dir = '/home/tsutar3/HEART/convos/SFT/empathy_10/twitter/'

new_convo = "/home/tsutar3/HEART/data/"

def check_data_size(data_dir):
    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Initialize a dictionary to store the sizes
    data_sizes = {}
    
    # Loop through each CSV file and get its size
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path)
        data_sizes[csv_file] = len(df)
    
    return data_sizes

if __name__ == "__main__":
    insta_data_sizes = check_data_size(new_convo)
    
    print("Insta Data Sizes:")
    for file, size in insta_data_sizes.items():
        print(f"{file}: {size} rows")


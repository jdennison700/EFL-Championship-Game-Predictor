import pandas as pd
import os

data = pd.read_csv(r"C:\Users\Joshua\Documents\Programming practice\Python\Machine Learning\Projects\EFL Championship\Raw Datasets\20162017.csv", encoding='latin1')

championship_data = data[data['Competition'].str.contains('Sky Bet Championship', regex=True)]


championship_data.drop('Attendance', axis=1, inplace=True)
championship_data.drop('Competition', axis=1, inplace=True)
championship_data.drop('Round', axis=1, inplace=True)
championship_data.rename(columns={'Date/Time': 'Date'}, inplace=True)

championship_data.insert(0, 'Match Number', range(1, 1 + len(championship_data)))
championship_data['Date'] = pd.to_datetime(championship_data['Date'],  format='mixed', dayfirst=True)




repo_dir = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.join(repo_dir, 'Datasets')

file_name = "championship-2016-GMTStandardTime.csv"
target_file_path = os.path.join(target_folder, file_name)

championship_data.to_csv(target_file_path, index=False)

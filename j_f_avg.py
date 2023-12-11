import argparse
import pandas as pd
import os
import json

# Function to calculate the average of a list of values
def calculate_average(values):
    return sum(values) / len(values)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Calculate averages of specified columns in CSV files.')
parser.add_argument('--anno0', help='Path to the first CSV file')
parser.add_argument('--anno1', help='Path to the second CSV file')
parser.add_argument('--anno2', help='Path to the third CSV file')
parser.add_argument('--anno3', help='Path to the fourth CSV file')
parser.add_argument('--result_path', help='Output CSV file name')
args = parser.parse_args()

# List to store the data frames from each CSV file
data_frames = []

# Read each CSV file and store the data frame in the list
for file_path in [args.anno0, args.anno1, args.anno2, args.anno3]:
    df = pd.read_csv(os.path.join(file_path, 'global_results-val.csv'))
    data_frames.append(df)

# Extract columns for J&F-Mean, J-Mean, and F-Mean
# j_f_mean_values = [df['J&F-Mean'] for df in data_frames]
# j_mean_values = [df['J-Mean'] for df in data_frames]
# f_mean_values = [df['F-Mean'] for df in data_frames]
j_f_mean_values = [list(df['J&F-Mean'])[0] for df in data_frames]
j_mean_values = [list(df['J-Mean'])[0] for df in data_frames]
f_mean_values = [list(df['F-Mean'])[0] for df in data_frames]

# Calculate averages for J&F-Mean, J-Mean, and F-Mean
avg_j_f_mean = calculate_average(j_f_mean_values)
avg_j_mean = calculate_average(j_mean_values)
avg_f_mean = calculate_average(f_mean_values)

# Read CSV files 5 to 8 for m_iou values and calculate the average
m_iou_data_frames = [pd.read_csv(os.path.join(file_path, 'bbox_results-val.csv')) for file_path in [args.anno0, args.anno1, args.anno2, args.anno3]]
global_m_iou_values = [df[df['Sequence'] == 'Global']['m_iou'].iloc[0] for df in m_iou_data_frames]
avg_global_m_iou = calculate_average(global_m_iou_values)

# Output original and average values to a new file
# output_data = {
#     'J&F-Mean': j_f_mean_values,
#     'J-Mean': j_mean_values,
#     'F-Mean': f_mean_values,
#     'Avg_J&F-Mean': avg_j_f_mean,
#     'Avg_J-Mean': avg_j_mean,
#     'Avg_F-Mean': avg_f_mean,
#     # 'm_iou': list(m_iou_data_frames['m_iou']),
#     'Avg_m_iou': avg_global_m_iou,
# }
output_data = {
    'm_iou': [round(val, 5) for val in global_m_iou_values],
    'J&F-Mean': [round(val, 5) for val in j_f_mean_values],
    'J-Mean': [round(val, 5) for val in j_mean_values],
    'F-Mean': [round(val, 5) for val in f_mean_values],
    'Avg_Global_m_iou': round(avg_global_m_iou, 5),
    'Avg_J&F-Mean': round(avg_j_f_mean, 5),
    'Avg_J-Mean': round(avg_j_mean, 5),
    'Avg_F-Mean': round(avg_f_mean, 5),
}


with open(os.path.join(args.result_path, 'score.json'), 'w') as json_file:
    json.dump(output_data, json_file, indent=2)
print('Average J&F-Mean: {}'.format(avg_j_f_mean))
print('Done.')

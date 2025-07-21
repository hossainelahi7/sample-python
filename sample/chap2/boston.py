# import os
# import ISLP as islp
# file_path = os.path.abspath('Boston')
# data = islp.load_data(file_path)
# print(data.describe())


import pandas as pd
import ISLP as islp

# Load the Boston dataset
file_path = 'Boston.csv'
boston = pd.read_csv(file_path)

# Initialize a DataFrame to store predictor information
predictor_df = pd.DataFrame(columns=['response', 'varname', 'R2', 'R2_log', 'R2_quad', 'max_R2'])

# Iterate over columns, excluding 'chas' (column index 3)
for i, col in enumerate(boston.columns):
    if i != 3:  # Exclude 'chas'
        response = col
        # Get predictor information using ISLP's best_predictor function
        predictor_info = islp.best_predictor(boston, col).iloc[0, 1:]  # Exclude the second column
        # Append the information to the DataFrame
        predictor_df = pd.concat([predictor_df, pd.DataFrame([[response] + predictor_info.tolist()],
                                                             columns=predictor_df.columns)], ignore_index=True)

# Print the DataFrame with the best predictors
print(predictor_df)

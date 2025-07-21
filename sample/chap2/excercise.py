import pandas as pd
import os
import matplotlib.pyplot as plt

# Replace with the absolute path to your CSV file
file_path = os.path.abspath('College.csv')

college = pd.read_csv(file_path)
# college2 = pd.read_csv(file_path, index_col=0)
college3 = college.rename(columns={'Unnamed: 0': 'College'})
college3 = college3.set_index('College')

college = college3

# print(college.describe())
# pd.plotting.scatter_matrix(df[['Top10perc','Apps','Enroll']])
# plt.show()

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Produce side-by-side boxplots of Outstate versus Private
college.boxplot(column='Outstate', by='Private', grid=False, ax=axes[0])
plt.title('Outstate vs Private')
plt.xlabel('Private')
plt.ylabel('Outstate')
# plt.show()

college['Elite'] = pd.cut(college['Top10perc'],
                           [0 , 50 ,100] ,
                           labels =['No', 'Yes'])

college.boxplot(column='Outstate', by='Elite', grid=False, ax=axes[1])
plt.title('Outstate vs Elite')
plt.xlabel('Elite')
plt.ylabel('Outstate')

plt.suptitle('')  # Suppress the default title
plt.tight_layout()
plt.show()

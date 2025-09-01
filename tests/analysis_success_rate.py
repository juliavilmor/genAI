import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results of the test success rate
#df = pd.read_csv('test_generation_dataset_autoreg/results_success_rate_dm256_nh4_ff1024_nl6_expanded.csv', index_col=0)
df = pd.read_csv('test_generation_new_dataset/results_success_rate_dm256_nh4_ff1024_nl6.csv', index_col=0)

# Compute the mean, median, mode, and standard deviation of the success rate
mean = df['success_rate'].mean()
median = df['success_rate'].median()
mode = df['success_rate'].mode()
std = df['success_rate'].std()

# Print the results
print('Mean:', mean)
print('Median:', median)
print('Mode:', mode)
print('Standard Deviation:', std)

# Plot the histogram of the success rate
plt.figure()
sns.histplot(df['success_rate'], bins=20, kde=True)
plt.xlabel('Success Rate')
plt.ylabel('Counts')
plt.title('Histogram of Success Rate')
plt.savefig('test_generation_new_dataset/histogram_success_rate_exp.png')

# Plot the success rate as a function of the weight index
plt.figure(figsize=(10,6), dpi=300)
sns.boxplot(x='weight_idx', y='success_rate', data=df)
plt.xlabel('Weight Index')
plt.ylabel('Success Rate')
plt.title('Success Rate as a Function of Weight Index')
plt.savefig('test_generation_new_dataset/success_rate_vs_weight_index_exp.png')

# Plot the histogram of the success rate for each weight index
plt.figure()
sns.histplot(df, x='success_rate', hue='weight_idx', bins=30, element='step', kde=True, linewidth=0)
plt.xlabel('Success Rate')
plt.ylabel('Counts')
plt.title('Histogram of Success Rate for Each Weight Index')
plt.savefig('test_generation_new_dataset/histogram_success_rate_vs_weight_index_exp.png')

# Success rate per protein sequence
plt.figure(figsize=(30,10))
sns.boxplot(x='seq_idx', y='success_rate', data=df)
plt.xlabel('Sequence Index')
plt.ylabel('Success Rate')
plt.title('Success Rate per Protein Sequence')
plt.savefig('test_generation_new_dataset/success_rate_vs_sequence_index_exp.png')
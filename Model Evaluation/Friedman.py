import pandas as pd
from scipy.stats import friedmanchisquare
import numpy as np

# enter data
data = pd.read_csv(r'D:\Msc\ML\Projects\Εργασία 6η\algo_performance.csv')

# calculate friedman Test
statistics, pValue = friedmanchisquare(data['C4.5'], data['1-NN'], data['NaiveBayes'], data['Kernel'], data['CN2'])
print('Statistics = %.3f, p-Value = %s' % (statistics, pValue))

# Set alpha value
for alpha in np.arange(0, 0.05, 0.0001):
    if pValue > alpha:
        print(
            "For alpha = {0} --> Fail to reject Hypothesis ( They don't have statistically significant difference)".format(
                alpha))
    else:
        print("For alpha = {0} --> Reject Hypothesis ( They have statistically significant difference)".format(alpha))

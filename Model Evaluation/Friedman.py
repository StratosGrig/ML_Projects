import pandas as pd
from scipy.stats import friedmanchisquare

# enter data
data = pd.read_csv(r'D:\Msc\ML\Projects\Εργασία 6η\algo_performance.csv')
print(data)
# calculate friedman Test
statists, pValue = friedmanchisquare(data['C4.5'], data['1-NN'], data['NaiveBayes'], data['Kernel'], data['CN2'])
print((statists, pValue))

# Set alpha value
alpha = 0.05
if pValue > alpha:
    print('Fail to reject Hypothesis')
else:
    print('Reject Hypothesis')

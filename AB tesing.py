import numpy as np
import pandas as pd
from scipy import stats

num_users = int(input("Enter number of users: "))
conversion_rate = float(input("Enter conversion rate (0-1): "))

np.random.seed(42)

user_ids = np.arange(num_users)

assignments = np.random.choice(['control', 'test'], size=len(user_ids))

conversions = np.random.binomial(1, p=conversion_rate, size=len(user_ids))
revenue = conversions * np.random.uniform(100, 200, size=len(user_ids))

data = pd.DataFrame({
    'user_id': user_ids,
    'group': assignments,
    'conversion': conversions,
    'revenue': revenue
})

grouped_data = data.groupby('group').agg({
    'conversion': 'mean',
    'revenue': 'mean'
}).reset_index()

conversion_test = stats.ttest_ind(
    data[data['group'] == 'control']['conversion'],
    data[data['group'] == 'test']['conversion']
)

revenue_test = stats.ttest_ind(
    data[data['group'] == 'control']['revenue'],
    data[data['group'] == 'test']['revenue']
)

print(grouped_data)
print(f"Conversion Rate Test: p-value = {conversion_test.pvalue}")
print(f"Revenue Test: p-value = {revenue_test.pvalue}")

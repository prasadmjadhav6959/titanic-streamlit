import pandas as pd
import ast

# Parse predictions.log
data = []
with open('predictions.log', 'r') as f:
    for line in f:
        if 'Input:' in line:
            input_str = line.split('Input: ')[1].split(', Prediction:')[0]
            input_dict = ast.literal_eval(input_str)
            # Extract first item from each list
            row = {k: v[0] for k, v in input_dict.items()}
            data.append(row)

# Save to CSV
pd.DataFrame(data).to_csv('current_inputs.csv', index=False)
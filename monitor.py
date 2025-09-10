import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from alibi_detect.cd import TabularDrift
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess reference (training) data
ref_data = pd.read_csv('train.csv')
ref_data = ref_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
ref_data['Age'].fillna(ref_data['Age'].median(), inplace=True)
ref_data['Embarked'].fillna(ref_data['Embarked'].mode()[0], inplace=True)
le_sex = LabelEncoder()
ref_data['Sex'] = le_sex.fit_transform(ref_data['Sex'])
le_embarked = LabelEncoder()
ref_data['Embarked'] = le_embarked.fit_transform(ref_data['Embarked'])

# Load current data (from logged predictions)
# Convert predictions.log to CSV manually or via script
try:
    current_data = pd.read_csv('current_inputs.csv')
except FileNotFoundError:
    print("Create current_inputs.csv from predictions.log")
    exit()

# Define categorical columns (indices)
cat_cols = [ref_data.columns.get_loc(col) for col in ['Sex', 'Embarked', 'Pclass']]

# Initialize drift detector
cd = TabularDrift(
    x_ref=ref_data.values,
    p_val=0.05,
    categories_per_feature={i: None for i in cat_cols}
)

# Detect drift
preds = cd.predict(current_data.values)
print("Drift detected" if preds['data']['is_drift'] else "No drift detected")
with open('drift_report.txt', 'w') as f:
    f.write(f"Drift: {preds['data']['is_drift']}, p-value: {preds['data']['p_val']}")
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Load cleaned data
df = pd.read_csv('streamlit-apps/house-price/train.csv')

# Select features
selected_features = [
    'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'SalePrice',
    'HouseStyle'
]

df = df[selected_features]

# One-hot encode HouseStyle
df = pd.get_dummies(df, columns=['HouseStyle'], drop_first=True)

# Drop rows with missing values (just to be safe)
df.dropna(inplace=True)

# Split into X and y
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('streamlit-apps/house-price/house_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as house_model.pkl")

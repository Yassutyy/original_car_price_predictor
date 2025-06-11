import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the data (adjust separator if tab-separated)
df = pd.read_csv('car_data.csv', sep='\t')

# Ensure correct columns
df.columns = ['Brand', 'Year', 'Selling_Price', 'KM_Driven', 'Fuel']

# Encode Categorical Features
le_brand = LabelEncoder()
le_fuel = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Fuel'] = le_fuel.fit_transform(df['Fuel'])

# Final features
X = df[['Brand', 'Year', 'KM_Driven', 'Fuel']]
y = df['Selling_Price']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and encoders
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('brand_encoder.pkl', 'wb') as f:
    pickle.dump(le_brand, f)
with open('fuel_encoder.pkl', 'wb') as f:
    pickle.dump(le_fuel, f)

"""Step 1: look at the skincare data before building anything."""
import pandas as pd

df = pd.read_csv('data/cosmetics.csv')

print('Rows (products):', len(df))
print('Columns:', list(df.columns))

print('\nFirst 3 products:')
print(df[['Label', 'Brand', 'Name', 'Price', 'Rank']].head(3).to_string())

print('\nProducts of each type:')
print(df['Label'].value_counts().to_string())

print('\nPrice: cheapest', df['Price'].min(), '| most expensive', df['Price'].max(),
      '| typical (median)', df['Price'].median())

print('\nIngredients of the first product:')
print(df['Ingredients'].iloc[0][:300], '...')
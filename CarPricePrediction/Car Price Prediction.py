import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from difflib import get_close_matches

# Load dataset
df = pd.read_csv("used_cars.csv")

# Prepare columns
df['Name'] = df['brand'].astype(str) + ' ' + df['model'].astype(str)
df['New_Price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['New_Price'] = df['New_Price'] * 83  # USD to INR conversion
df['Year'] = pd.to_numeric(df['model_year'], errors='coerce')
df.dropna(subset=['New_Price', 'Year'], inplace=True)

# Clean name and encode
df['Name_clean'] = df['Name'].str.strip().str.lower()
le = LabelEncoder()
df['Name_encoded'] = le.fit_transform(df['Name_clean'])

# Features & Target
X = df[['Name_encoded', 'Year']]
y = df['New_Price']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Format INR nicely
def format_inr(value):
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value / 1e5:.2f} Lakh"
    else:
        return f"₹{value:,.0f}"

# Store predictions
predictions_list = []

# Prediction loop
while True:
    print("-- Car Price Prediction Menu --")
    print(" 1. Car Name & Year ")
    print(" 2. Save to CSV")
    print(" 3. Exit\n")

    choice = input("Enter your choice : ").strip()

    if choice == '1':
        user_name = input("Enter Car Name (brand + model): ").strip().lower()
        try:
            user_year = int(input("Enter Year of Purchase: "))
        except ValueError:
            print("Enter valid numeric year\n.")
            continue

        car_names = df['Name_clean'].unique()
        match = get_close_matches(user_name, car_names, n=1, cutoff=0.6)

        if match:
            matched = match[0]
            encoded = df[df['Name_clean'] == matched]['Name_encoded'].values[0]
            car_row = df[df['Name_clean'] == matched]
            original_name = car_row['Name'].values[0]

            if user_year in car_row['Year'].values:
                pred_input = pd.DataFrame([[encoded, user_year]], columns=['Name_encoded', 'Year'])
                pred_price = model.predict(pred_input)[0]
                print(f"\nCar: {original_name} ({user_year})")
                print(f"Predicted Price: {format_inr(pred_price)}")

                predictions_list.append({
                    "Car Name": original_name,
                    "Year": user_year,
                    "Predicted Price (INR)": format_inr(pred_price)
                })
            else:
                valid_years = sorted(int(y) for y in car_row['Year'].unique())
                print(f"\n'{original_name}' not sold in {user_year}.")
                print(f"Available years: {valid_years}")
        else:
            print("Car not found. Please check spelling.\n")

    elif choice == '2':
        if predictions_list:
            pd.DataFrame(predictions_list).to_csv("predicted_prices.csv", index=False)
            print("Predictions saved to 'predicted_prices.csv'.")
        else:
            print("No predictions to save.\n")

    elif choice == '3':
        print("Goodbye! Thanks for using the predictor.")
        break

    else:
        print("Invalid choice. Please enter 1, 2, or 3.\n")

from google.colab import files
files.download("predicted_prices.csv")
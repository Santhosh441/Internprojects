import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

Details=[]

# Load the dataset
a = pd.read_csv("/content/output.csv")
x = a[['bedrooms', 'sqft_lot']]
y = a['price']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
m = LinearRegression()
m.fit(x_train, y_train)

# Evaluate model
y_pred = m.predict(x_test)
r2 = r2_score(y_test, y_pred)

# Prediction function with input validation
def predict():
    try:
        w = float(input("\nEnter BHK: "))
        u = float(input("Enter Sqft: "))

        if w <= 0 or u <= 0:
            print("BHK and Sqft must be positive numbers.\n")
            return

        s = pd.DataFrame([[w, u]], columns=['bedrooms', 'sqft_lot'])
        p = m.predict(s)

        print(f"Price: ₹{p[0]:,.2f}")
        print(f"Accuracy: {r2 * 10000:.2f}%\n")

        Details.append({
            "Bedrooms": w,
            "Sqft": u,
            "Predicted Price": round(p[0], 2)
        })

    except ValueError:
        print("Invalid input. Please enter numeric values only.\n")

# Save function
def savecon():
    if Details:
        df = pd.DataFrame(Details)
        df.to_csv("House_Predictions.csv", index=False)
        print("Predictions saved to 'House_Predictions.csv'\n")
    else:
        print("No predictions to save.\n")

# Main loop
while True:
    print("-- Welcome To House Price Prediction --")
    print("1. Predict Price")
    print("2. Save to CSV File")
    print("3. Exit")

    a = input("\nEnter Choice (1/2/3): ").strip()

    if a == "1":
        predict()
    elif a == "2":
        savecon()
    elif a == "3":
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.\n")

from google.colab import files
files.download("House_Predictions.csv")
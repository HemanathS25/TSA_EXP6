# Ex.No: 6               HOLT WINTERS METHOD

### Date: 28/08/2026

### Dataset : BMW SALES(2010-2024)

### AIM:

To implement the Holt Winters Method Model using Python.

### ALGORITHM:

1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
    
### PROGRAM:

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv(
    "/content/BMW sales data (2010-2024).csv"
)

data.columns = data.columns.str.strip()

data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Sales_Volume'] = pd.to_numeric(
    data['Sales_Volume'],
    errors='coerce'
)

data = data.dropna(
    subset=['Year', 'Sales_Volume']
)

data_yearly = data.groupby(
    'Year'
)['Sales_Volume'].sum()

data_yearly.index = pd.to_datetime(
    data_yearly.index.astype(int).astype(str),
    format='%Y'
)

data_yearly = data_yearly.sort_index()

print("BMW Sales Dataset:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Columns:")
print(data.columns)

print("\nYearly BMW Sales Data:")
print(data_yearly)

plt.figure(figsize=(10, 5))
data_yearly.plot()
plt.title('BMW Yearly Sales')
plt.xlabel('Year')
plt.ylabel('Sales Volume')
plt.show()

scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        data_yearly.values.reshape(-1, 1)
    ).flatten(),
    index=data_yearly.index
)

plt.figure(figsize=(10, 5))
scaled_data.plot()
plt.title('Scaled BMW Sales Data')
plt.xlabel('Year')
plt.ylabel('Scaled Sales')
plt.show()

decomposition = seasonal_decompose(
    data_yearly,
    model='additive',
    period=3
)

decomposition.plot()
plt.show()

scaled_data = scaled_data + 1

train_data = scaled_data[
    :int(len(scaled_data) * 0.8)
]

test_data = scaled_data[
    int(len(scaled_data) * 0.8):
]

print("\nTraining Data:")
print(train_data)

print("\nTest Data:")
print(test_data)

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=3
).fit()

test_predictions_add = model_add.forecast(
    steps=len(test_data)
)

ax = train_data.plot(
    figsize=(10, 5)
)

test_predictions_add.plot(
    ax=ax
)

test_data.plot(
    ax=ax
)

ax.legend([
    "train_data",
    "test_predictions_add",
    "test_data"
])

ax.set_title('Visual Evaluation')
ax.set_xlabel('Year')
ax.set_ylabel('Scaled Sales')

plt.show()

rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_add
    )
)

mae = mean_absolute_error(
    test_data,
    test_predictions_add
)

print("\nModel Performance Metrics:")
print("RMSE:", rmse)
print("MAE:", mae)

standard_deviation = np.sqrt(
    scaled_data.var()
)

mean_value = scaled_data.mean()

print("\nStandard Deviation and Mean:")
print("Standard Deviation:", standard_deviation)
print("Mean:", mean_value)

final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=3
).fit()

final_predictions = final_model.forecast(
    steps=3
)

predictions_for_inverse = (
    final_predictions - 1
)

final_predictions_original = scaler.inverse_transform(
    predictions_for_inverse.values.reshape(-1, 1)
).flatten()

future_dates = pd.date_range(
    start=data_yearly.index[-1] + pd.offsets.YearBegin(1),
    periods=3,
    freq='YS'
)

final_predictions_original = pd.Series(
    final_predictions_original,
    index=future_dates
)

ax = data_yearly.plot(
    figsize=(10, 5)
)

final_predictions_original.plot(
    ax=ax
)

ax.legend([
    "Actual BMW Sales",
    "Predicted BMW Sales"
])

ax.set_xlabel('Year')
ax.set_ylabel('Sales Volume')
ax.set_title('BMW Sales Prediction')

plt.show()

print("\nFinal BMW Sales Predictions:")
print(final_predictions_original)
```

### OUTPUT:

# TEST_PREDICTION

<img width="719" height="392" alt="image" src="https://github.com/user-attachments/assets/343a53db-f9ea-4da3-b1da-e28511f91bf8" />

<img width="728" height="549" alt="image" src="https://github.com/user-attachments/assets/51242b76-0b53-4615-9950-d22c7a725330" />


# FINAL_PREDICTION

<img width="993" height="541" alt="image" src="https://github.com/user-attachments/assets/f3ffdfc4-42e9-4fe7-af27-597b29d6fb02" />


### RESULT:

Thus the program run successfully based on the Holt Winters Method model.

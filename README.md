# 🌤️ Deep Learning Weather Forecaster | Kolkata 7-Day Prediction System

A deep learning system that predicts the daily temperature sequence for the next week using Bidirectional LSTMs and historical weather data.

## 📌 Project Overview

This project implements an intelligent weather forecasting engine designed specifically for the tropical climate of Kolkata, India. Using advanced Recurrent Neural Networks (Bi-LSTM) and 5 years of historical data from the [Five Years Weather Data of Kolkata](https://www.kaggle.com/datasets/kafkarps/five-years-weather-data-of-kolkata) dataset, the system analyzes 14-day weather patterns to generate a precise 7-day temperature forecast, achieving sub-1-degree accuracy.

## ✨ Features

  * **7-Day Sequence Prediction:** Forecasts temperature for Day 1 to Day 7 simultaneously (not just tomorrow).
  * **Bi-Directional Context:** Utilizes Bidirectional LSTMs to understand weather patterns from both past and future contexts.
  * **Robust Outlier Handling:** Implements Huber Loss to maintain stability during sudden storms or heatwaves.
  * **Seasonality Engineering:** Uses cyclical sine/cosine features to mathematically model Winter, Summer, and Monsoon cycles.
  * **Data Cleaning Pipeline:** Removal of noisy features (Wind Speed/Chill) and missing value imputation.
  * **Visual Feedback:** Plots "Actual vs. Predicted" temperature trends for easy verification.

## 🎯 How It Works

### 1\. Data Preprocessing

  * **Source:** 5 years of daily records (2017–2022) for Kolkata.
  * **Feature Selection:** Filters for 11 key signals (Temperature, Dew Point, Pressure, etc.).
  * **Normalization:** Scales all values to 0-1 range for efficient Neural Network training.

### 2\. Sequence Creation

  * **Input Window:** Captures the past **14 days** of weather history.
  * **Target Window:** Aligns data to predict the subsequent **7 days**.
  * **Sliding Window:** Moves one day at a time to create thousands of training examples.

### 3\. Deep Learning Model (Pro Model)

  * **Architecture:** Bidirectional LSTM (64 units) → Dropout (0.3) → LSTM (32 units).
  * **Learning:** Trains on historical sequences to minimize Huber Loss.
  * **Optimization:** Uses the Adam optimizer with adaptive learning rate reduction (`ReduceLROnPlateau`).

### 4\. Visual Output

  * **Console:** Prints the 7-day temperature forecast array (in °C).
  * **Graph:** Generates a line chart comparing the predicted trend against actual historical values for validation.

## 🛠️ Technical Specifications

  * **Model:** Bidirectional LSTM (TensorFlow/Keras)
  * **Input Horizon:** 14 Days
  * **Forecast Horizon:** 7 Days
  * **Accuracy:** 0.99°C Mean Absolute Error (MAE)
  * **Loss Function:** Huber Loss (delta=1.0)
  * **Training Time:** \~20-30 Epochs (with Early Stopping)

## 📦 Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

**Required Libraries:**

  * Python 3.x
  * TensorFlow 2.x
  * Pandas & NumPy
  * Scikit-Learn
  * Matplotlib

## 🚀 Installation & Usage

### 1\. Clone the Repository

```bash
git clone https://github.com/trishpurkait/weather-forecaster-kolkata.git
cd weather-forecaster-kolkata
```

### 2\. Install Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

### 3\. Run the Training Notebook

Open and run `deep-learning-weather-forecaster-kolkata(1).ipynb` to train the model from scratch.

### 4\. Make a Prediction

Use the saved model to forecast weather for new data:

```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_weather_pro.keras')

# Predict next 7 days (input must be shape [1, 14, 11])
prediction = model.predict(recent_weather_data)
print("7-Day Forecast:", scaler.inverse_transform(prediction))
```

## 📊 System Output

**Console Output:**

```text
Training Days: 1826
Testing Days: 365
Final Accuracy: +/- 0.99 °C
Standard Model Max Error: 5.78 °C
Pro Model Max Error:      5.58 °C
```

**Visual Output:**

  * **Loss Plot:** Training vs. Validation loss curves.
  * **Forecast Plot:** A graph showing the predicted 7-day temperature curve overlaying the actual ground truth.
  * **Error Histogram:** A bell curve showing most errors are centered around 0°C.

## ⚙️ Configuration Parameters

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `LOOKBACK_WINDOW` | 14 | Days of history input to the model |
| `FORECAST_HORIZON` | 7 | Days of future prediction output |
| `BATCH_SIZE` | 32 | Samples processed per training step |
| `DROPOUT` | 0.3 | Regularization rate to prevent overfitting |
| `EPOCHS` | 100 | Maximum training iterations |

## 🔧 Troubleshooting

**Issue: Model accuracy is low (\> 2°C error)**

  * Ensure input data is correctly scaled (0-1).
  * Check if `Day_sin` and `Day_cos` columns were created correctly for seasonality.

**Issue: Predictions are flat/constant**

  * The model might be "underfitting." Increase the LSTM units or train for more epochs.
  * Verify that `Wind Chill` (empty column) was removed.

## 🎯 Use Cases

  * **Event Planning:** Selecting optimal dates for outdoor weddings or sports.
  * **Agriculture:** Planning irrigation schedules based on heat forecasts.
  * **Supply Chain:** Anticipating demand for beverages or AC units.
  * **Personal Use:** Planning travel or commute for the upcoming week.

## 🚀 Future Enhancements

  * [ ] **Ensemble Learning:** Combine predictions from 3 models to reduce error further.
  * [ ] **Web App Integration:** Deploy using Streamlit for a live dashboard.
  * [ ] **Rainfall Classification:** Add a separate model to predict "Rain / No Rain."
  * [ ] **Transformer Integration:** Experiment with Temporal Fusion Transformers (TFT).

## 📧 Contact

**Trish Purkait**

  * GitHub: [@trishpurkait](https://www.google.com/search?q=https://github.com/trishpurkait)
  * Email: trishpurkait@gmail.com

## 🙏 Acknowledgments

  * [Five Years Weather Data of Kolkata](https://www.kaggle.com/datasets/kafkarps/five-years-weather-data-of-kolkata) (Kaggle)
  * TensorFlow/Keras Community
  * Open Source Contributors

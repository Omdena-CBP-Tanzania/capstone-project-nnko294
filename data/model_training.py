{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9ae940d0-31f3-4ee8-9792-092f86b7df4d",
   "metadata": {},
   "source": [
    "# Project: Climate Change Analysis in Tanzania"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "374782d2-8678-49f6-99a2-27b46683f4fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install --upgrade pip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c38157e8-5f7f-4fbf-97d2-9c74bb324ef2",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install pandas numpy matplotlib seaborn scikit-learn statsmodels joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "522d4be1-f20f-4def-b00d-1373727fcbe2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Data Collection\n",
    "# Import necessary libraries\n",
    "# Data Handling\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import os\n",
    "import warnings\n",
    "from datetime import datetime\n",
    "\n",
    "# Visualization\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Time Series Decomposition\n",
    "from statsmodels.tsa.seasonal import seasonal_decompose\n",
    "\n",
    "# Machine Learning\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
    "\n",
    "# Suppress warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "92195a69-e4bf-48e6-802c-cab5397c989d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Year</th>\n",
       "      <th>Month</th>\n",
       "      <th>Average_Temperature_C</th>\n",
       "      <th>Total_Rainfall_mm</th>\n",
       "      <th>Max_Temperature_C</th>\n",
       "      <th>Min_Temperature_C</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2000</td>\n",
       "      <td>1</td>\n",
       "      <td>26.1</td>\n",
       "      <td>19.8</td>\n",
       "      <td>32.0</td>\n",
       "      <td>21.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2000</td>\n",
       "      <td>2</td>\n",
       "      <td>25.8</td>\n",
       "      <td>87.3</td>\n",
       "      <td>29.5</td>\n",
       "      <td>22.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>26.8</td>\n",
       "      <td>266.5</td>\n",
       "      <td>29.9</td>\n",
       "      <td>21.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2000</td>\n",
       "      <td>4</td>\n",
       "      <td>26.3</td>\n",
       "      <td>136.7</td>\n",
       "      <td>30.1</td>\n",
       "      <td>22.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2000</td>\n",
       "      <td>5</td>\n",
       "      <td>26.0</td>\n",
       "      <td>63.1</td>\n",
       "      <td>30.7</td>\n",
       "      <td>22.4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Year  Month  Average_Temperature_C  Total_Rainfall_mm  Max_Temperature_C  \\\n",
       "0  2000      1                   26.1               19.8               32.0   \n",
       "1  2000      2                   25.8               87.3               29.5   \n",
       "2  2000      3                   26.8              266.5               29.9   \n",
       "3  2000      4                   26.3              136.7               30.1   \n",
       "4  2000      5                   26.0               63.1               30.7   \n",
       "\n",
       "   Min_Temperature_C  \n",
       "0               21.9  \n",
       "1               22.7  \n",
       "2               21.8  \n",
       "3               22.9  \n",
       "4               22.4  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the climate dataset\n",
    "# Dataset should be previously downloaded and saved as 'tanzania_climate_data.csv'\n",
    "df = pd.read_csv(r\"D:\\TANZANIA_KIC\\capstone-project-nnko294\\data\\tanzania_climate_data.csv\")\n",
    "\n",
    "# Preview the dataset\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fe6eb90e-5ced-4a46-bd11-f6a67272d5c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Data Preprocessing\n",
    "\n",
    "# Handle missing values \n",
    "df = df.dropna().sum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "82709768-9388-4ec0-acf7-2fec5b5697ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert date columns to datetime format\n",
    "df['Date'] = pd.to_datetime(df['Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60b00128-215f-4bf5-a2f6-1359980157ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature engineering: extract year, month, and season\n",
    "df['Year'] = df['Date'].dt.year\n",
    "df['Month'] = df['Date'].dt.month\n",
    "df['Season'] = df['Month'].map({12: 'Wet', 1: 'Wet', 2: 'Wet',\n",
    "                                 3: 'Dry', 4: 'Dry', 5: 'Dry',\n",
    "                                 6: 'Cool', 7: 'Cool', 8: 'Cool',\n",
    "                                 9: 'Hot', 10: 'Hot', 11: 'Hot'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77fff9e1-0492-4d01-b742-e9424105d51d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding categorical variables (if applicable)\n",
    "df = pd.get_dummies(df, columns=['Season'], drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05e3c511-439b-4874-8f93-a9336211a94b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Exploratory Data Analysis (EDA)\n",
    "# --------------------------------------------\n",
    "\n",
    "# Descriptive statistics\n",
    "print(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f020b198-deaa-4a2d-9a71-b0fa03fc16f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot temperature trend over time\n",
    "plt.figure(figsize=(10,5))\n",
    "plt.plot(df['Date'], df['Temperature'], label='Temperature')\n",
    "plt.title('Temperature Trend Over Time')\n",
    "plt.xlabel('Year')\n",
    "plt.ylabel('Temperature (°C)')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88a7d7ad-f79f-431f-ae89-6709a5ec1128",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot precipitation trend\n",
    "plt.figure(figsize=(10,5))\n",
    "plt.plot(df['Date'], df['Precipitation'], color='green', label='Precipitation')\n",
    "plt.title('Precipitation Trend Over Time')\n",
    "plt.xlabel('Year')\n",
    "plt.ylabel('Precipitation (mm)')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c79e69ef-0dbd-40db-9fc3-5c9b502a6d71",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation heatmap\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(df.corr(), annot=True, cmap='coolwarm')\n",
    "plt.title('Correlation Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fc81be1b-5a97-442b-893d-2d344ac0f4eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Seasonal decomposition (requires datetime index)\n",
    "# Example decomposition for temperature\n",
    "temp_series = df.set_index('Date')['Temperature'].resample('M').mean()\n",
    "decomposition = seasonal_decompose(temp_series.dropna(), model='additive')\n",
    "decomposition.plot()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c4283cc6-cd61-49aa-9f40-8984990e2c37",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Machine Learning Model Development\n",
    "# --------------------------------------------\n",
    "\n",
    "# Define features and target\n",
    "features = df.drop(columns=['Date', 'Temperature'])\n",
    "target = df['Temperature']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36f4f49d-27a0-42c5-b65e-843b363340cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d033c95b-8c9d-48ad-bd05-c069400257ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train a Random Forest model\n",
    "rf = RandomForestRegressor(random_state=42)\n",
    "rf.fit(X_train, y_train)\n",
    "y_pred_rf = rf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2352dc6f-e175-4ed8-a61b-731f40f0ac12",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate model\n",
    "print(\"Random Forest RMSE:\", np.sqrt(mean_squared_error(y_test, y_pred_rf)))\n",
    "print(\"Random Forest MAE:\", mean_absolute_error(y_test, y_pred_rf))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f72e2c5c-9717-42e0-9ea7-7ce992756b6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the trained model for deployment\n",
    "joblib.dump(rf, 'random_forest_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79700e18-71e5-44ad-86a2-1192873f21c0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

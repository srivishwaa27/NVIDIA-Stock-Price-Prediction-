NVIDIA Stock Price Prediction Using Machine Learning and Deep Learning
This project aims to forecast the daily closing price of NVIDIA Corporation’s stock using a combination of machine learning and deep learning techniques. Leveraging 10 years of historical stock data (from January 2014 to March 2024), the project applies advanced time series modeling and evaluation techniques to identify the most accurate prediction models.

Project Objectives
Predict the next-day closing price of NVIDIA stock using historical data.
Implement and compare models such as LSTM, Bi-LSTM, Prophet, and Random Forest.
Evaluate models using multiple metrics including RMSE, MAE, MAPE, and R² Score.
Forecast future stock prices and visualize trends using interactive and static charts.

Dataset Description
The dataset consists of 10 years of daily trading data for NVIDIA Corporation, with key attributes including:

Date, Open, High, Low, Close
Daily_Return (percentage change)
Rolling_Volatility (30-day standard deviation)
ATR (Average True Range)
Rolling_Mean, Upper_Band, Lower_Band (Bollinger Bands)

Preprocessing Workflow
Converted the Date column to datetime format.
Handled missing values by dropping null records.
Detected and removed outliers using Z-Score and IQR methods.
Applied MinMax Scaling for deep learning models to stabilize training.
Split the dataset into time-based training and testing sets (80/20).

Exploratory Data Analysis (EDA)
Box plots to visualize outliers in closing prices.
Z-score distribution plots to assess data distribution and anomalies.
30-day Rolling Volatility Plot to observe trend stability over time.
Candlestick Charts for year-wise price movement visualization.
Correlation Heatmap to understand feature relationships.

Models Used
Model	Description
LSTM	Used 60-day historical windows to predict the next day's closing price.
Bi-LSTM	Captures both past and future dependencies in price movements.
Prophet	Decomposes time series into trend and seasonality; good for interpretable forecasting.
Random Forest	Tree-based ensemble regressor that performs exceptionally on structured data.

Model Evaluation
Model	RMSE	MAE	MAPE	R² Score
LSTM	1.08	0.73	32.74%	0.9940
Bi-LSTM	1.99	1.36	–	0.9859
Prophet	4.45	2.41	–	0.8975
Random Forest	0.28	0.14	–	0.9996

Random Forest achieved the best performance with an R² Score of 0.9996, indicating nearly perfect predictions.

Future Forecasting
Using the trained Bi-LSTM model, the project performs a 10-day future price prediction, simulating real-world short-term trading insights. This highlights the practical application of deep learning in financial forecasting.

🛠️ Tools & Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn, TensorFlow/Keras, Facebook Prophet
Jupyter Notebook / Google Colab
Machine Learning: Random Forest
Deep Learning: LSTM, Bi-LSTM
Visualization: Plotly, Candlestick Charts, Heatmaps

Conclusion
This project demonstrates the effectiveness of combining traditional and deep learning models for financial forecasting. The low error rates and high R² values validate the accuracy and reliability of the approach. Future enhancements could include adding macroeconomic indicators, implementing attention-based models, and using hyperparameter tuning techniques for further optimization.

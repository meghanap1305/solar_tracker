Solar Pro Max: AI-Powered Solar Energy Planner
Overview: Solar Pro Max is a solar planning tool that goes beyond simple averages. Instead of just looking at last year's weather, this application uses a Deep Learning (LSTM) model to analyze 40 years of historical NASA data and forecast solar irradiance for the next 5 years. It combines this AI prediction with a robust financial model to tell you exactly how much money a solar setup will save you over 25 years, accounting for inflation and hardware degradation.
How It Works (The AI Part)
Most solar calculators just take an average of past sunlight. I took a different approach because weather is a Time-Series problem.
Data Source: The app fetches roughly 40 years of monthly solar radiation (GHI) data from the NASA POWER API for your specific latitude/longitude.
The Model: I built a Long Short-Term Memory (LSTM) neural network using PyTorch. Standard neural networks treat data points individually, but LSTMs have "memory cells" that can remember long-term patterns (like seasonal cycles and multi-year climate trends), making them perfect for weather forecasting.
Training: The model looks at a sliding window of the past 24 months to learn how to predict the next month.
Forecasting: Once trained, it autoregressively predicts the next 60 months (5 years) to give a realistic baseline for energy generation.
Financial Modeling
The app doesn't just calculate Power x Price. It simulates the real-world lifecycle of a solar plant
It assumes panels lose 0.5% efficiency every year due to wear and tear.
It calculates future savings based on rising electricity tariffs (Energy Inflation).
ROI Analysis: It compares three different system configurations ("Basic," "Premium," "Beast Mode") to find the best payback period.
Tech Stack
Frontend: Streamlit (Python) for the interactive dashboard and plotting.
AI/ML: PyTorch (building and running the LSTM).
Data Processing: NumPy & Pandas for handling time-series arrays.
Visualization: Matplotlib for plotting the "History vs. Future" prediction curves.
API: NASA POWER API (JSON).
File Structure
solarapp.py: The main frontend application. Handles the UI, user inputs, and financial math.
solarmodel.py: The AI backend. Defines the SimpleLSTM class, handles data normalization, and runs the prediction loop.
solar_lstm_temp.pt: The pre-trained PyTorch model weights. This ensures the app runs instantly without needing to retrain the neural network every time.
(this is tagged as releases due to the file size)

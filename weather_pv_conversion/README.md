# Project Overview

This folder, `weather_pv_conversion`, contains the codebase developed for predicting solar photovoltaic (PV) power generation using machine learning techniques. The project is centered around three main components: a Jupyter Notebook for experimentation and development, a Python module for operational deployment, and a directory containing the serialized model artifacts.

The notebook, `solar_pv_prediction_notebook.ipynb`, documents the complete workflow—from data acquisition to PV system simulation, model training, and performance evaluation. It served as the main environment for iterative development and validation. Once the best predictive model was identified, its logic was encapsulated in the `solar_predictor.py` script through a reusable class named `SolarProductionPredictor`, designed for integration into a broader application. The trained model and preprocessing tools are stored in the `output/model/` directory, which includes both the serialized model (`best_estimator_model.pkl`) and the feature scaler (`scaler.pkl`) used during training to ensure consistency in prediction.

# Objective

The main objective of this work is to develop an accurate and efficient tool for predicting solar PV energy production, which is essential for optimizing energy management strategies and supporting decision-making in renewable energy systems. Instead of relying solely on detailed physical simulations, the project aims to build a data-driven model that approximates AC power output from a PV system using weather forecast data. This approach enables faster predictions and easier integration into systems such as those used for sizing and configuring solar panels and batteries.

# Workflow and Implementation

The methodology is implemented and described step-by-step in the notebook. First, weather data is retrieved from the Open-Meteo API for a specified location. The retrieved data includes key atmospheric parameters such as solar irradiance (GHI, DHI, DNI), temperature, wind speed, and cloud cover. To ensure reproducibility and reduce repeated API calls during development, responses are cached locally.

The `pvlib` library was employed to simulate the hourly AC power output of a defined PV system. The simulation was based on components such as Canadian Solar Inc. CS6X-300M panels and an SMA SB5000TL inverter, with configuration parameters like tilt and azimuth taken into account. The `ModelChain` functionality of `pvlib` allowed us to generate reliable output data from the weather inputs, which was then used as the target variable for training the machine learning model.

For the predictive model, a Support Vector Regressor (SVR) was chosen due to its ability to handle non-linear relationships effectively. The training data was prepared using `pandas`, and input features were normalized using `StandardScaler` from `scikit-learn`. Hyperparameters for the SVR were tuned using `GridSearchCV`, exploring various combinations of `C`, `gamma`, `kernel`, and `degree`. Once the optimal configuration was identified, the model was evaluated using standard regression metrics such as the R² score, and its generalization capability was tested on unseen data. The trained SVR model and the corresponding scaler were then serialized using `joblib` and saved for future use.

To operationalize the solution, the `solar_predictor.py` module was created. It defines the `SolarProductionPredictor` class, which automates the prediction workflow. Upon initialization, the class loads the trained model and scaler. When provided with a time range, it fetches new weather data from the Open-Meteo API, preprocesses the data to match the training format, and uses the SVR model to predict the AC power output. The results are returned as a `pandas` DataFrame indexed by datetime, and a utility method is provided to compute the total production over a specified period.


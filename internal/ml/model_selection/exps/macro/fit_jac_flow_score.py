from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("base_dir", "../exp_data")
from src.query_api.query_api_img import ImgScoreQueryApi
from src.query_api.query_api_mlp import GTMLP
from src.common.constant import Config, CommonVars
from src.query_api.query_api_img import Gt201, Gt101

GTMLP(Config.Criteo)
GTMLP(Config.Frappe)
GTMLP(Config.UCIDataset)
all_tabulars = GTMLP._instances

api = Gt201()
ImgScoreQueryApi(Config.NB201, Config.c10)
ImgScoreQueryApi(Config.NB201, Config.imgNet)
ImgScoreQueryApi(Config.NB201, Config.c100)
all_image = ImgScoreQueryApi._instances

data = {
    "nas_wot": [],
    "synflow": [],
    "labels": []
}

# for key, value in all_tabulars.items():
#     for arch_id in value.mlp_score.keys():
#         acc, _ = value.get_valid_auc(str(arch_id), 100)
#         naswot_score = value.api_get_score(str(arch_id))[CommonVars.NAS_WOT]
#         synflow_score = value.api_get_score(str(arch_id))[CommonVars.PRUNE_SYNFLOW]
#         data["nas_wot"].append(naswot_score)
#         data["synflow"].append(synflow_score)
#         data["labels"].append(acc)

for key, value in all_image.items():
    dataset = key[1]
    for arch_id in value.data.keys():
        try:
            acc, _ = api.query_200_epoch(arch_id, dataset)
            naswot_score = float(value.api_get_score(str(arch_id))[CommonVars.NAS_WOT])
            synflow_score = float(value.api_get_score(str(arch_id))[CommonVars.PRUNE_SYNFLOW])
            data["nas_wot"].append(naswot_score)
            data["synflow"].append(synflow_score)
            data["labels"].append(acc)
        except Exception as e:
            print(e)


def fit_linear_regression(X, y):
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # Mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))
    # Coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination (R^2): %.2f'
          % r2_score(y_test, y_pred))


def fit_polynomial(X, y):
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create linear regression object
    regr = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),  # Add polynomial features
        ('linear', LinearRegression())  # Fit a linear model
    ])

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # Access the linear model to get the coefficients
    linear_step = regr.named_steps['linear']  # Access the 'linear' step of the pipeline
    coefficients = linear_step.coef_  # Get the coefficients (weights)
    intercept = linear_step.intercept_  # Get the intercept

    # The coefficients
    print('Coefficients: \n', coefficients)

    # Mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

    # Coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination (R^2): %.2f' % r2_score(y_test, y_pred))

# Prepare feature matrix 'X' and labels 'y'
X = np.array([[f1_i, f2_i] for f1_i, f2_i in zip(data['nas_wot'], data['synflow'])])
y = np.array(data['labels'])
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

fit_polynomial(X_normalized, y)

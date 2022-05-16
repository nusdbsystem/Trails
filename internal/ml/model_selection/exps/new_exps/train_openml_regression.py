import os
import random
import time
import warnings
import torch
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# this corresponds to the number of threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_data(
        task_id: int,
        val_share: float = 0.25,
        test_size: float = 0.2,
        seed: int = 11):
    task = openml.tasks.get_task(task_id=task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute)

    # Automatically identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['uint8', 'uint32', 'uint64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    print("dataset = ", X.shape, "column = ", X.shape[1])

    if numerical_cols.shape[0] + categorical_cols.shape[0] != X.shape[1]:
        print("Errored: Not all columns are detected")
        exit(0)
    if isinstance(y[1], bool):
        y = y.astype('bool')

    print("All labels", set(y.tolist()))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    print("train_dataset = ", X_train.shape, "test_dataset = ", X_test.shape)

    resampling_strategy_args = {'val_share': val_share}
    return X_train, X_test, y_train, y_test, resampling_strategy_args, categorical_indicator, numerical_cols, categorical_cols


seed = 11

# year
# task_id = 361091

# California Housing
# task_id = 361089

# pool
task_id = 361034


if __name__ == '__main__':
    # Setting up reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    number_of_configurations_limit = 0
    start_time = time.time()
    X_train, X_test, y_train, y_test, resampling_strategy_args, \
    categorical_indicator, numerical_cols, categorical_cols = get_data(
        task_id=task_id,
        seed=seed)

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # model = "384-256-256-512"
    # model = "256-384-512-512"
    model = "384-256-128-256"
    corrected_tuple_value = tuple(int(num) for num in model.split('-'))
    print(corrected_tuple_value)
    mlp = MLPRegressor(hidden_layer_sizes=corrected_tuple_value,
                       max_iter=500,
                       activation='relu',
                       solver='adam',
                       random_state=60)

    # Define transformers for numerical and categorical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Combine transformers in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create a pipeline with a scaler and the MLPClassifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('mlp', mlp)  # MLP model
    ])

    # Use the pipeline for training
    pipeline.fit(X_train, y_train)

    # Predicting the Test set results
    from sklearn.metrics import mean_squared_error, r2_score

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'MLP Mean Squared Error: {mse}, R-squared: {r2}, rmse: {rmse}')

    """
    XGBoost Mean Squared Error: 0.01920498835787958, R-squared: 0.8486075354525808
    CatBoost Mean Squared Error: 0.017099762514787762, R-squared: 0.8652029804940158
    """

    # exit(0)

    import xgboost as xgb

    # xgboost_model
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', seed=seed)
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgboost', xgboost_model)
    ])

    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)

    print(f'XGBoost Mean Squared Error: {mse_xgb}, R-squared: {r2_xgb}, rmse_xgb: {rmse_xgb}')

    import catboost as cb

    # catboost_model
    catboost_model = cb.CatBoostRegressor(random_seed=seed, verbose=0)
    cb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('catboost', catboost_model)
    ])

    cb_pipeline.fit(X_train, y_train)
    y_pred_cb = cb_pipeline.predict(X_test)
    mse_cb = mean_squared_error(y_test, y_pred_cb)
    r2_cb = r2_score(y_test, y_pred_cb)
    rmse_cgb = np.sqrt(mse_cb)

    print(f'CatBoost Mean Squared Error: {mse_cb}, R-squared: {r2_cb}, rmse_cgb: {rmse_cgb}')

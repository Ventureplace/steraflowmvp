import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
import logging
from typing import Dict, List, Optional
from io import BytesIO
from datetime import datetime

# External libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import klib
from fuzzywuzzy import process

# Configure logging
logging.basicConfig(
    filename='data_processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Constants
DEFAULT_CONFIG_PATH = 'config.json'

# Load configuration
def load_config(config_path: Optional[str] = None) -> Dict:
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    else:
        logging.warning("No configuration file found. Using default settings.")
        return {
            "schema_mapping": {
                "timestamp": ["timestamp", "date_recorded", "shipment_time", "order_datetime", "date", "time"],
                "id": ["product_id", "product_code", "item_id", "sku", "order_number"]
            },
            "missing_value_handling": {
                "method": "drop"  # Options: "drop", "mean", "mode", "interpolate"
            },
            "outlier_handling": {
                "method": "remove",  # Options: "remove", "cap"
                "threshold": 3  # Z-score threshold
            },
            "data_normalization": {
                "method": None  # Options: "standard", "minmax", None
            },
            "feature_engineering": {
                "enabled": False,
                "lag_columns": [],
                "lag_periods": [1, 7, 30],
                "rolling_columns": [],
                "rolling_windows": [7, 30, 90]
            },
            "anomaly_detection": {
                "z_score_threshold": 3
            },
            "fuzzy_matching": {
                "threshold": 80
            }
        }

# Data preprocessing functions (from your second code snippet)
def handle_missing_values(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    method = config.get("missing_value_handling", {}).get("method", "drop")
    logging.info(f"Handling missing values with method: {method}")

    if method == "drop":
        df = df.dropna()
        logging.info("Dropped all rows with missing values.")
    elif method == "mean":
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        logging.info("Imputed missing values in numerical columns with column means.")
    elif method == "mode":
        df = df.fillna(df.mode().iloc[0])
        logging.info("Imputed missing values with column modes.")
    else:
        logging.warning(f"Unknown missing value handling method: {method}. No changes made.")

    return df

def handle_outliers(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    method = config.get("outlier_handling", {}).get("method", "remove")
    threshold = config.get("outlier_handling", {}).get("threshold", 3)
    logging.info(f"Handling outliers with method: {method}, threshold: {threshold}")

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in num_cols:
        z_scores = stats.zscore(df[col])
        if method == "remove":
            df = df[(z_scores < threshold) & (z_scores > -threshold)]
            logging.info(f"Removed outliers from column: {col}")
        elif method == "cap":
            df[col] = df[col].clip(
                lower=df[col].quantile(0.05),
                upper=df[col].quantile(0.95)
            )
            logging.info(f"Capped outliers in column: {col}")
        else:
            logging.warning(f"Unknown outlier handling method: {method}. No changes made to column: {col}")

    return df

def normalize_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    method = config.get("data_normalization", {}).get("method")
    logging.info(f"Normalizing data with method: {method}")

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if num_cols.empty:
        logging.warning("No numerical columns found for normalization. Skipping normalization step.")
        return df

    if method == "standard":
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logging.info("Applied StandardScaler to numerical columns.")
    elif method == "minmax":
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logging.info("Applied MinMaxScaler to numerical columns.")
    elif method is None:
        logging.info("No normalization applied.")
    else:
        logging.warning(f"Unknown normalization method: {method}. No changes made.")

    return df

def detect_anomalies(df: pd.DataFrame, config: Dict) -> None:
    logging.info("Starting anomaly detection...")

    z_score_threshold = config.get('anomaly_detection', {}).get('z_score_threshold', 3)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in num_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[z_scores > z_score_threshold]
        if not outliers.empty:
            logging.warning(f"Outliers detected in column '{col}': {len(outliers)} data points")
            logging.info(f"Sample outliers in '{col}': {outliers[col].head().tolist()}")

    logging.info("Anomaly detection completed.")

def create_correlation_plots(df: pd.DataFrame, output_dir: str) -> None:
    try:
        # Positive correlations
        plt.figure(figsize=(12, 10))
        klib.corr_plot(df, split='pos')
        plt.title("Positive Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'positive_correlations.png'))
        plt.close()

        # Negative correlations
        plt.figure(figsize=(12, 10))
        klib.corr_plot(df, split='neg')
        plt.title("Negative Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'negative_correlations.png'))
        plt.close()

        # Missing values plot
        plt.figure(figsize=(12, 10))
        klib.missingval_plot(df)
        plt.title("Missing Values")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values.png'))
        plt.close()

        logging.info("Created and saved correlation and missing values plots.")
    except Exception as e:
        logging.error(f"Failed to create plots: {e}")

# Streamlit app functions
def show(project_name):
    st.title(f"Data Cleaning for Project: {project_name}")

    if project_name not in st.session_state.projects:
        st.warning(f"Project '{project_name}' not found.")
        return

    if 'data' not in st.session_state.projects[project_name] or st.session_state.projects[project_name]['data'] is None:
        st.warning("No data available. Please upload data first.")
        return

    data = st.session_state.projects[project_name]['data']

    if not isinstance(data, pd.DataFrame):
        st.error("The data is not in the correct format. Please upload a valid dataset.")
        return

    # Load configuration
    config = load_config()

    st.subheader("Original Data Sample")
    st.write(data.head())

    # Data cleaning options
    st.subheader("Data Cleaning Options")

    # Handle missing values
    handle_missing = st.checkbox("Handle Missing Values")
    if handle_missing:
        missing_method = st.selectbox("Missing Value Handling Method", ["Drop", "Mean", "Mode"])
        config['missing_value_handling']['method'] = missing_method.lower()

    # Remove duplicates
    remove_duplicates = st.checkbox("Remove Duplicate Rows")
    if remove_duplicates:
        data = data.drop_duplicates()
        logging.info("Removed duplicate rows.")

    # Handle outliers
    handle_outliers_option = st.checkbox("Handle Outliers")
    if handle_outliers_option:
        outlier_method = st.selectbox("Outlier Handling Method", ["Remove", "Cap"])
        threshold = st.number_input("Z-score Threshold", value=3)
        config['outlier_handling']['method'] = outlier_method.lower()
        config['outlier_handling']['threshold'] = threshold

    # Normalize data
    normalize_data_option = st.checkbox("Normalize Data")
    if normalize_data_option:
        normalization_method = st.selectbox("Normalization Method", ["Standard", "MinMax"])
        config['data_normalization']['method'] = normalization_method.lower()

    # Feature Engineering
    feature_engineering_option = st.checkbox("Enable Feature Engineering")
    if feature_engineering_option:
        config['feature_engineering']['enabled'] = True
        lag_columns = st.multiselect("Columns for Lag Features", data.columns.tolist())
        lag_periods = st.multiselect("Lag Periods", [1, 7, 30], default=[1])
        rolling_columns = st.multiselect("Columns for Rolling Features", data.columns.tolist())
        rolling_windows = st.multiselect("Rolling Windows", [7, 30, 90], default=[7])
        config['feature_engineering']['lag_columns'] = lag_columns
        config['feature_engineering']['lag_periods'] = lag_periods
        config['feature_engineering']['rolling_columns'] = rolling_columns
        config['feature_engineering']['rolling_windows'] = rolling_windows

    # Apply cleaning when button is clicked
    if st.button("Apply Data Cleaning"):
        # Clean column names
        data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
        logging.info("Cleaned column names")

        # Handle missing values
        if handle_missing:
            data = handle_missing_values(data, config)
            logging.info("Handled missing values.")

        # Handle outliers
        if handle_outliers_option:
            data = handle_outliers(data, config)
            logging.info("Handled outliers.")

        # Normalize data
        if normalize_data_option:
            data = normalize_data(data, config)
            logging.info("Normalized data.")

        # Detect anomalies
        detect_anomalies(data, config)

        # Feature engineering
        if feature_engineering_option:
            data = feature_engineering(data, config)
            logging.info("Performed feature engineering.")

        # Save cleaned data
        st.session_state.projects[project_name]['cleaned_data'] = data
        st.success("Data cleaning applied and saved successfully!")

        # Generate plots
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)
        create_correlation_plots(data, output_dir)
        st.success("Correlation and missing values plots generated.")

    # Display cleaned data
    if 'cleaned_data' in st.session_state.projects[project_name]:
        st.subheader("Cleaned Data Sample")
        st.write(st.session_state.projects[project_name]['cleaned_data'].head())

# Feature Engineering Functions
def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    for col in columns:
        for window in windows:
            df[f'{col}_rolling_{window}'] = df[col].rolling(window=window).mean()
    return df

def feature_engineering(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    feature_config = config.get('feature_engineering', {})

    if feature_config.get('enabled', False):
        logging.info("Starting feature engineering...")

        # Ensure DataFrame is sorted by timestamp if exists
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        # Create lag features
        lag_columns = feature_config.get('lag_columns', [])
        lag_periods = feature_config.get('lag_periods', [1, 7, 30])
        if lag_columns:
            df = create_lag_features(df, lag_columns, lag_periods)
            logging.info(f"Created lag features for columns: {lag_columns}")

        # Create rolling average features
        rolling_columns = feature_config.get('rolling_columns', [])
        rolling_windows = feature_config.get('rolling_windows', [7, 30, 90])
        if rolling_columns:
            df = create_rolling_features(df, rolling_columns, rolling_windows)
            logging.info(f"Created rolling average features for columns: {rolling_columns}")

        logging.info("Feature engineering completed.")
    else:
        logging.info("Feature engineering skipped (disabled in config).")

    return df

if __name__ == "__main__":
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
    if 'current_project' not in st.session_state:
        st.session_state.current_project = "Default Project"
    show(st.session_state.current_project)
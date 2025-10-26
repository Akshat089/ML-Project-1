import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

# --- Configuration ---
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SUBMISSION_FILE = 'sample_submission.csv'
OUTPUT_FILE = 'submission_elasticnet.csv'

N_SPLITS = 10
RANDOM_SEED = 42

# ----------------- Data Loading -----------------
def load_data():
    print("--- 1. Loading Data ---")
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        sample_sub = pd.read_csv(SUBMISSION_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df, sample_sub

# ----------------- EDA -----------------
def perform_eda(train_df):
    print("--- 2. Performing EDA ---")
    original_skew = train_df['HotelValue'].skew()
    log_skew = np.log1p(train_df['HotelValue']).skew()
    print(f"Target Skew: Original={original_skew:.2f}, Log-Transformed={log_skew:.2f}")
    print("We'll use log-transformed target for training.\n")

# ----------------- Preprocessing -----------------
def preprocess_and_feature_engineer(train_df, test_df):
    print("--- 3. Preprocessing (ColumnTransformer + Imputer + OneHot + Scaler) ---")

    # Save test IDs
    test_ids = test_df['Id']

    # Log-transform target
    y_train = np.log1p(train_df['HotelValue'])

    # Drop target and ID
    X_train = train_df.drop(['Id', 'HotelValue'], axis=1)
    X_test = test_df.drop(['Id'], axis=1)

    # Identify column types
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    print(f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

    # Define preprocessing pipeline (same as before)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    # Add variance threshold selector
    selector = VarianceThreshold(0.01)

    # Combine preprocessing steps
    preprocessing_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector)
    ])

    # Fit on train, transform both
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    print(f"After preprocessing: Train shape = {X_train_processed.shape}, Test shape = {X_test_processed.shape}\n")

    return X_train_processed, X_test_processed, y_train, test_ids

# ----------------- Model Training -----------------
def train_model(X, y_train, X_test):
    print("--- 4. Model Training: Elastic Net Regression ---")

    # Elastic Net Model (mix of L1 & L2)
    model = ElasticNet(
        alpha=0.1,       # overall regularization strength
        l1_ratio=0.5,    # balance between L1 (Lasso) and L2 (Ridge)
        max_iter=5000,
        random_state=RANDOM_SEED
    )

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(X_test.shape[0])

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        val_preds = model.predict(X_val_fold)
        oof_preds[val_idx] = val_preds
        test_preds += model.predict(X_test) / N_SPLITS

        rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
        print(f"Fold {fold + 1} RMSE: {rmse:.5f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nOverall OOF RMSE (log-transformed): {overall_rmse:.5f}\n")

    return test_preds

# ----------------- Submission -----------------
def create_submission(test_ids, test_predictions):
    print("--- 5. Creating Submission ---")
    final_predictions = np.expm1(test_predictions)
    final_predictions[final_predictions < 0] = 0

    submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission file saved as: {OUTPUT_FILE}")
    print(submission_df.head())

# ----------------- Main Pipeline -----------------
if __name__ == "__main__":
    print("Starting Hotel Value Prediction Pipeline (Elastic Net)...")
    train_df, test_df, sample_sub = load_data()

    if train_df is not None:
        perform_eda(train_df)
        X, X_test, y_train, test_ids = preprocess_and_feature_engineer(train_df, test_df)
        test_predictions = train_model(X, y_train, X_test)
        create_submission(test_ids, test_predictions)
        print("\nPipeline finished successfully!")

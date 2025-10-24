import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import os

# --- Configuration ---
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")

# Define file names
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SUBMISSION_FILE = 'sample_submission.csv'
OUTPUT_FILE = 'submission.csv'

# K-Fold setup
N_SPLITS = 2
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

# ----------------- Preprocessing & Feature Engineering -----------------
def preprocess_and_feature_engineer(train_df, test_df):
    print("--- 3. Preprocessing & Feature Engineering ---")
    test_ids = test_df['Id']
    ntrain = len(train_df)

    # Log-transform target
    y_train = np.log1p(train_df['HotelValue'])

    # Drop Id and target
    train_df = train_df.drop(['Id', 'HotelValue'], axis=1)
    test_df = test_df.drop('Id', axis=1)

    # Combine train and test
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    print(f"Combined data shape: {all_data.shape}")

    # Fill missing numerical values
    num_cols_fill_zero = ['FacadeArea', 'BasementFacilitySF1', 'BasementFacilitySF2', 'BasementUnfinishedSF',
                          'BasementTotalSF', 'BasementFullBaths', 'BasementHalfBaths', 'ParkingArea',
                          'SwimmingPoolArea', 'ExtraFacilityValue', 'ParkingConstructionYear']
    for col in num_cols_fill_zero:
        all_data[col] = all_data[col].fillna(0)
    all_data['RoadAccessLength'] = all_data['RoadAccessLength'].fillna(all_data['RoadAccessLength'].median())

    # Fill missing categorical values
    none_cols = ['ServiceLaneType', 'FacadeType', 'BasementHeight', 'BasementCondition', 'BasementExposure',
                 'BasementFacilityType1', 'BasementFacilityType2', 'ParkingType', 'ParkingFinish', 'ParkingQuality',
                 'ParkingCondition', 'PoolQuality', 'BoundaryFence', 'ExtraFacility', 'LoungeQuality']
    for col in none_cols:
        all_data[col] = all_data[col].fillna('None')
    mode_cols = ['ZoningCategory', 'UtilityAccess', 'ElectricalSystem', 'KitchenQuality', 'PropertyFunctionality']
    for col in mode_cols:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # Ordinal mapping
    quality_mapping = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    ordinal_cols = ['ExteriorQuality','ExteriorCondition','BasementHeight','BasementCondition',
                    'HeatingQuality','KitchenQuality','LoungeQuality','ParkingQuality','ParkingCondition','PoolQuality']
    for col in ordinal_cols:
        all_data[col] = all_data[col].map(quality_mapping).fillna(0)
    all_data['BasementExposure'] = all_data['BasementExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}).fillna(0)
    all_data['PropertyFunctionality'] = all_data['PropertyFunctionality'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7}).fillna(7)
    all_data['ParkingFinish'] = all_data['ParkingFinish'].map({'None':0,'Unf':1,'RFn':2,'Fin':3}).fillna(0)

    # Feature engineering
    all_data['PropertyAge'] = all_data['YearSold'] - all_data['ConstructionYear']
    all_data['AgeSinceRemodel'] = all_data['YearSold'] - all_data['RenovationYear']
    all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
    all_data['TotalBaths'] = all_data['FullBaths'] + 0.5*all_data['HalfBaths'] + all_data['BasementFullBaths'] + 0.5*all_data['BasementHalfBaths']
    all_data['TotalPorchSF'] = all_data['TerraceArea'] + all_data['OpenVerandaArea'] + all_data['EnclosedVerandaArea'] + all_data['SeasonalPorchArea'] + all_data['ScreenPorchArea']
    all_data['HasPool'] = (all_data['SwimmingPoolArea']>0).astype(int)
    all_data['OverallQuality_Cond'] = all_data['OverallQuality']*all_data['OverallCondition']

    print(f"Feature engineering complete. New data shape: {all_data.shape}\n")

    X = all_data[:ntrain]
    X_test = all_data[ntrain:]

    return X, X_test, y_train, test_ids

# ----------------- One-Hot Encoding for GBM -----------------
def preprocess_for_gbm(X, X_test):
    X_all = pd.concat([X, X_test], axis=0)
    X_all = pd.get_dummies(X_all, drop_first=True)
    X = X_all.iloc[:len(X), :]
    X_test = X_all.iloc[len(X):, :]
    print(f"After one-hot encoding, train shape: {X.shape}, test shape: {X_test.shape}\n")
    return X, X_test

# ----------------- Model Training -----------------
def train_model(X, y_train, X_test):
    print("--- 4. Model Training: Gradient Boosting Regressor ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_predictions = np.zeros(X.shape[0])
    test_predictions = np.zeros(X_test.shape[0])

    for fold, (train_index, val_index) in enumerate(kf.split(X, y_train)):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model = GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            max_features=0.8,
            random_state=RANDOM_SEED
        )

        model.fit(X_train_fold, y_train_fold)
        oof_predictions[val_index] = model.predict(X_val_fold)
        test_predictions += model.predict(X_test) / N_SPLITS

    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    print(f"\nTraining complete. OOF RMSE: {oof_rmse:.5f}\n")
    return test_predictions

# ----------------- Submission -----------------
def create_submission(test_ids, test_predictions):
    print("--- 5. Creating Submission ---")
    final_predictions = np.expm1(test_predictions)
    final_predictions[final_predictions<0] = 0
    submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission file saved at {OUTPUT_FILE}")
    print(submission_df.head())

# ----------------- Main Pipeline -----------------
if __name__ == "__main__":
    print("Starting Hotel Value Prediction Pipeline...")
    train_df, test_df, sample_sub = load_data()
    if train_df is not None:
        perform_eda(train_df)
        X, X_test, y_train, test_ids = preprocess_and_feature_engineer(train_df, test_df)
        X, X_test = preprocess_for_gbm(X, X_test)
        test_predictions = train_model(X, y_train, X_test)
        create_submission(test_ids, test_predictions)
        print("\nPipeline finished successfully!")

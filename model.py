import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
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
OUTPUT_FILE = 'submission.csv'

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

# ----------------- Feature Engineering -----------------
def feature_engineering(df):
    df = df.copy()
    
    # --- Basement Features ---
    basement_quality_map = {'GLQ':5,'ALQ':4,'BLQ':3,'Rec':3,'LwQ':2,'Unf':1,'None':0}
    df['BasementFacilitySF1'] = df.get('BasementFacilitySF1',0).fillna(0)
    df['BasementFacilitySF2'] = df.get('BasementFacilitySF2',0).fillna(0)
    df['Type1_Score'] = df.get('BasementFacilityType1','None').map(basement_quality_map).fillna(0)
    df['Type2_Score'] = df.get('BasementFacilityType2','None').map(basement_quality_map).fillna(0)
    df['TotalBasementScore'] = df['Type1_Score']*df['BasementFacilitySF1'] + df['Type2_Score']*df['BasementFacilitySF2']
    df['BasementFinishedSF'] = df['BasementFacilitySF1'] + df['BasementFacilitySF2']
    df.drop(columns=['BasementFacilityType1','BasementFacilityType2','BasementFacilitySF1','BasementFacilitySF2','Type1_Score','Type2_Score'], errors='ignore', inplace=True)
    
    # --- Pool Features ---
    pool_quality_map = {'None':0,'Fa':1,'Ex':2}
    df['SwimmingPoolArea'] = df.get('SwimmingPoolArea',0).fillna(0)
    df['PoolQuality'] = df.get('PoolQuality','None').fillna('None')
    df['TotalPoolScore'] = df['SwimmingPoolArea']*df['PoolQuality'].map(pool_quality_map).fillna(0)
    df.drop(columns=['SwimmingPoolArea','PoolQuality'], errors='ignore', inplace=True)
    
    # --- Porch Features ---
    df['TotalPorchArea'] = df.get('OpenVerandaArea',0).fillna(0) + df.get('EnclosedVerandaArea',0).fillna(0) + df.get('SeasonalPorchArea',0).fillna(0) + df.get('ScreenPorchArea',0).fillna(0)
    df.drop(columns=['OpenVerandaArea','EnclosedVerandaArea','SeasonalPorchArea','ScreenPorchArea'], errors='ignore', inplace=True)
    
    # --- Ordinal Mappings ---
    quality_map_5pt = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}
    parking_finish_map = {'Fin':3,'RFn':2,'Unf':1,'None':0}
    functionality_map = {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'None':0}
    exposure_map = {'Gd':4,'Av':3,'Mn':2,'No':1,'None':0}
    
    df['ParkingQuality'] = df.get('ParkingQuality','None').map(quality_map_5pt).fillna(0)
    df['ParkingCondition'] = df.get('ParkingCondition','None').map(quality_map_5pt).fillna(0)
    df['ParkingFinish'] = df.get('ParkingFinish','None').map(parking_finish_map).fillna(0)
    df['ExteriorQuality'] = df.get('ExteriorQuality','None').map(quality_map_5pt).fillna(0)
    df['ExteriorCondition'] = df.get('ExteriorCondition','None').map(quality_map_5pt).fillna(0)
    df['BasementCondition'] = df.get('BasementCondition','None').map(quality_map_5pt).fillna(0)
    df['BasementExposure'] = df.get('BasementExposure','None').map(exposure_map).fillna(0)
    df['KitchenQuality'] = df.get('KitchenQuality','None').map(quality_map_5pt).fillna(0)
    df['HeatingQuality'] = df.get('HeatingQuality','None').map(quality_map_5pt).fillna(0)
    df['PropertyFunctionality'] = df.get('PropertyFunctionality','None').map(functionality_map).fillna(0)
    
    # --- Time-based Features ---
    df['HouseAge'] = df.get('YearSold',0) - df.get('ConstructionYear',0)
    df['RenovationYear'] = df.get('RenovationYear', df.get('ConstructionYear',0))
    df.loc[df['RenovationYear']==0,'RenovationYear'] = df.loc[df['RenovationYear']==0,'ConstructionYear']
    df['YearsSinceModification'] = df['YearSold'] - df[['ConstructionYear','RenovationYear']].max(axis=1)
    
    # --- Interaction Features ---
    df['QualityArea'] = df.get('OverallQuality',0)*df.get('UsableArea',0)
    if 'FullBaths' in df.columns and 'HalfBaths' in df.columns:
        df['TotalBathrooms'] = df['FullBaths'] + 0.5*df['HalfBaths']
    
    # Log-transform skewed numerical features
    for col in ['RoadAccessLength','LandArea','FacadeArea','BasementTotalSF','ParkingArea']:
        if col in df.columns:
            df[col+'_Log'] = np.log1p(df[col].fillna(0))
    
    # Drop original columns used for engineered features
    df.drop(columns=['ConstructionYear','RenovationYear','YearSold','MonthSold'], errors='ignore', inplace=True)
    return df

# ----------------- Preprocessing -----------------
def preprocess_and_feature_engineer(train_df, test_df):
    print("--- 3. Preprocessing & Feature Engineering ---")

    # Save test IDs
    test_ids = test_df['Id']

    # Log-transform target
    y_train = np.log1p(train_df['HotelValue'])

    # Drop target and ID
    X_train = train_df.drop(['Id','HotelValue'], axis=1)
    X_test = test_df.drop(['Id'], axis=1)

    # Apply feature engineering
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    # Identify numeric/categorical
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object','category']).columns

    print(f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

    # Define preprocessing
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    selector = VarianceThreshold(0.01)
    preprocessing_pipeline = Pipeline([('preprocessor', preprocessor),('selector', selector)])

    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    print(f"After preprocessing: Train shape = {X_train_processed.shape}, Test shape = {X_test_processed.shape}\n")
    return X_train_processed, X_test_processed, y_train, test_ids

# ----------------- Model Training -----------------
def train_model(X, y_train, X_test):
    print("--- 4. Model Training: Lasso Regression ---")
    model = LassoCV(alphas=np.logspace(-4,1,50), cv=5, max_iter=5000, random_state=RANDOM_SEED)

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
        print(f"Fold {fold+1} RMSE: {rmse:.5f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nOverall OOF RMSE (log-transformed): {overall_rmse:.5f}\n")
    return test_preds

# ----------------- Submission -----------------
def create_submission(test_ids, test_predictions):
    print("--- 5. Creating Submission ---")
    final_predictions = np.expm1(test_predictions)
    final_predictions[final_predictions < 0] = 0
    submission_df = pd.DataFrame({'Id': test_ids,'HotelValue':final_predictions})
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission file saved as: {OUTPUT_FILE}")
    print(submission_df.head())

# ----------------- Main Pipeline -----------------
if __name__ == "__main__":
    print("Starting Hotel Value Prediction Pipeline (Lasso + FE)...")
    train_df, test_df, sample_sub = load_data()
    if train_df is not None:
        perform_eda(train_df)
        X, X_test, y_train, test_ids = preprocess_and_feature_engineer(train_df, test_df)
        test_predictions = train_model(X, y_train, X_test)
        create_submission(test_ids, test_predictions)
        print("\nPipeline finished successfully!")

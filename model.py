import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')
import xgboost as xgb

# ---------------- Configuration ----------------
TRAIN_FILE = 'train_folder/train.csv'
TEST_FILE = 'test_folder/test.csv'
SUBMISSION_FILE = 'submission.csv'
RANDOM_SEED = 42
N_SPLITS = 10

# ----------------- 1. Load Data -----------------
def load_data():
    print("--- 1. Loading Data ---")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df

# ----------------- 2. EDA -----------------
def perform_eda(train_df):
    print("--- 2. Performing EDA ---")
    print(f"Original target skew: {train_df['HotelValue'].skew():.2f}")
    print(f"Log-transformed target skew: {np.log1p(train_df['HotelValue']).skew():.2f}")
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(train_df['HotelValue'], kde=True, bins=50, color='skyblue')
    plt.title("HotelValue Distribution (Original)")
    
    plt.subplot(1,2,2)
    sns.histplot(np.log1p(train_df['HotelValue']), kde=True, bins=50, color='orange')
    plt.title("HotelValue Distribution (Log-Transformed)")
    plt.show()
    
    # --- Missing Data ---
    missing_pct = (train_df.isnull().sum() / len(train_df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print(f"Top 5 Features with Missing Data:")
    print(missing_pct.head(6))
    
    plt.figure(figsize=(12,6))
    sns.heatmap(train_df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Missing Data Heatmap")
    plt.show()

    # --- Correlation Matrix for Numerical Features ---
    numeric_cols = train_df.select_dtypes(include=np.number).columns
    corr_matrix = train_df[numeric_cols].corr()
    
    plt.figure(figsize=(15,12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.show()
    
    # --- Top Features Correlated with HotelValue (Top 10) ---
    top_corr_features = corr_matrix['HotelValue'].sort_values(ascending=False).drop('HotelValue').head(10)
    print("Top 10 Numerical Features Correlated with HotelValue:")
    print(top_corr_features)

# Bar plot of correlation values
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_corr_features.values, y=top_corr_features.index, palette="viridis")
    plt.title("Top 10 Features Most Correlated with HotelValue")
    plt.xlabel("Correlation with HotelValue")
    plt.ylabel("Feature")
    plt.show()

# Plot each top feature against HotelValue
    for feature in top_corr_features.index:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=train_df[feature], y=train_df['HotelValue'])
        plt.title(f"{feature} vs HotelValue")
        plt.xlabel(feature)
        plt.ylabel("HotelValue")
        plt.show()
    

    
    
    print("\nEDA complete with graphs.\n")

# ----------------- 3. Preprocessing & Feature Engineering -----------------
def preprocess_and_feature_engineer(train_df, test_df):
    print("--- 3. Preprocessing & Feature Engineering ---")

    # Save test IDs
    test_ids = test_df['Id']
    
    # Log-transform target
    y_train = np.log1p(train_df['HotelValue'])
    
    # Drop target and Id
    X_train = train_df.drop(['Id', 'HotelValue'], axis=1)
    X_test = test_df.drop(['Id'], axis=1)
    
    # ---------------- Outlier Removal ----------------
    lower, upper = y_train.quantile(0.001), y_train.quantile(0.999)
    mask = (y_train >= lower) & (y_train <= upper)
    if 'UsableArea' in X_train.columns:
        mask &= (X_train['UsableArea'] < 4000)
    if 'OverallQuality' in X_train.columns and 'UsableArea' in X_train.columns:
        mask &= ~((X_train['OverallQuality'] < 3) & (X_train['UsableArea'] > 3000))
    X_train = X_train[mask]
    y_train = y_train[mask]

    # ---------------- Basement Features ----------------
    basement_map = {'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0}
    pool_map = {'None':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4} 
    for df in [X_train, X_test]:
        df['QualityRating'] = df['OverallQuality']  # create new column
        df['ValueDensity'] = df['QualityRating'] / np.log1p(df['LandArea'].fillna(0) + 1)
    # Basement
        df['BasementArea1'] = df['BasementFacilitySF1'].fillna(0)
        df['BasementArea2'] = df['BasementFacilitySF2'].fillna(0)
        df['BasementQualityScore'] = df['BasementFacilityType1'].map(basement_map).fillna(0) + \
                                 df['BasementFacilityType2'].map(basement_map).fillna(0)
        df['BasementImpact'] = (df['BasementArea1'] + df['BasementArea2']) * df['BasementQualityScore']
        df.drop(columns=['BasementFacilitySF1','BasementFacilitySF2','BasementFacilityType1','BasementFacilityType2'], inplace=True)

    # Pool
        df['PoolArea'] = df['SwimmingPoolArea'].fillna(0)
        df['PoolQualityScore'] = df['PoolQuality'].map(pool_map).fillna(0)
        df['PoolImpact'] = df['PoolArea'] * df['PoolQualityScore']
        df.drop(columns=['SwimmingPoolArea','PoolQuality'], inplace=True)

    # Porch
        df['PorchSpace'] = df[['OpenVerandaArea','EnclosedVerandaArea','SeasonalPorchArea','ScreenPorchArea']].fillna(0).sum(axis=1)
        df.drop(columns=['OpenVerandaArea','EnclosedVerandaArea','SeasonalPorchArea','ScreenPorchArea'], inplace=True)

    # House age
        df['AgeSinceRenovation'] = df['YearSold'] - df[['ConstructionYear','RenovationYear']].max(axis=1)

    # Interaction
        df['ValueDensity'] = df['QualityRating'] / np.log1p(df['LandArea'].fillna(0) + 1)


    # Drop unused columns
    drop_cols = ['Id','BoundaryFence','ExtraFacility','ServiceLaneType','BasementHalfBaths','LowQualityArea','FacadeType','ParkingArea']
    X_train.drop(columns=drop_cols, errors='ignore', inplace=True)
    X_test.drop(columns=drop_cols, errors='ignore', inplace=True)

    print(f"Final training shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")
    return X_train, X_test, y_train, test_ids

# ----------------- 4. Model Training -----------------
def train_model(X, y, X_test):
    print("--- 4. Model Training: Linear Regression ---")
    
    # Separate numeric and categorical columns
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    # Linear Regression model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Fit model
    model.fit(X, y)
    y_test_pred_log = model.predict(X_test)
    return y_test_pred_log

# ----------------- 5. Submission -----------------
def create_submission(test_ids, y_pred_log):
    y_pred = np.expm1(y_pred_log)
    y_pred[y_pred<0] = 0
    submission = pd.DataFrame({'Id': test_ids, 'HotelValue': y_pred})
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")
    print(submission.head())

# ----------------- Main -----------------
if __name__=="__main__":
    train_df, test_df = load_data()
    perform_eda(train_df)
    X_train, X_test, y_train, test_ids = preprocess_and_feature_engineer(train_df, test_df)
    y_test_pred_log = train_model(X_train, y_train, X_test)
    create_submission(test_ids, y_test_pred_log)

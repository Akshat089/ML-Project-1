import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
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
N_SPLITS = 10
RANDOM_SEED = 42

def load_data():
    """Loads train, test, and submission files from the root directory."""
    print("--- 1. Loading Data ---")
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        sample_sub = pd.read_csv(SUBMISSION_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure train.csv, test.csv, and sample_submission.csv are in the same directory as this script.")
        return None, None, None
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print("Data loaded successfully.\n")
    return train_df, test_df, sample_sub

def perform_eda(train_df):
    """Performs and prints key EDA insights with graphs."""
    print("--- 2. Performing EDA ---")
    
    # --- Target Variable Analysis ---
    original_skew = train_df['HotelValue'].skew()
    log_skew = np.log1p(train_df['HotelValue']).skew()
    
    print(f"Target Variable (HotelValue) Skewness:")
    print(f"  Original: {original_skew:.2f} (Highly right-skewed)")
    print(f"  Log-Transformed: {log_skew:.2f} (Much closer to normal)")
    print("Conclusion: We will use a log-transformed target for training.\n")
    
    # --- Target Distribution Graphs ---
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
    print(missing_pct.head(5))
    
    # Missing Data Heatmap
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
    
    # --- Top Features Correlated with HotelValue ---
    top_corr_features = corr_matrix['HotelValue'].sort_values(ascending=False).head(6)[1:]
    print("Top 5 Numerical Features Correlated with HotelValue:")
    print(top_corr_features)
    
    # Pairplot for top correlated features (optional, can be slow on large datasets)
    top_features = top_corr_features.index.tolist() + ['HotelValue']
    sns.pairplot(train_df[top_features])
    plt.show()
    
    # --- Categorical Features Distributions ---
    cat_cols = train_df.select_dtypes(include='object').columns
    for col in cat_cols[:5]:  # Show first 5 for brevity
        plt.figure(figsize=(8,4))
        sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()
    
    print("\nEDA complete with graphs.\n")


def preprocess_and_feature_engineer(train_df, test_df):
    """Cleans data, imputes missing values, and creates new features."""
    print("--- 3. Preprocessing & Feature Engineering ---")
    
    # Store test IDs for submission and ntrain for splitting
    test_ids = test_df['Id']
    ntrain = len(train_df)

    # **Log-transform the target variable**
    y_train = np.log1p(train_df['HotelValue'])

    # Drop the original target and Id column
    train_df = train_df.drop(['Id', 'HotelValue'], axis=1)
    test_df = test_df.drop('Id', axis=1)

    # **Combine train and test data for consistent preprocessing**
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    print(f"Combined data shape: {all_data.shape}")

    # --- Handle Missing Data ---
    
    # Numerical features: Fill with 0 (for areas, counts) or median
    for col in ['FacadeArea', 'BasementFacilitySF1', 'BasementFacilitySF2', 'BasementUnfinishedSF', 'BasementTotalSF',
                'BasementFullBaths', 'BasementHalfBaths', 'ParkingArea', 'SwimmingPoolArea', 'ExtraFacilityValue',
                'ParkingConstructionYear']:
        all_data[col] = all_data[col].fillna(0)
    
    all_data['RoadAccessLength'] = all_data['RoadAccessLength'].fillna(all_data['RoadAccessLength'].median())

    # Categorical features: Fill with 'None' or mode
    none_cols = ['ServiceLaneType', 'FacadeType', 'BasementHeight', 'BasementCondition', 'BasementExposure',
                 'BasementFacilityType1', 'BasementFacilityType2', 'ParkingType', 'ParkingFinish', 'ParkingQuality',
                 'ParkingCondition', 'PoolQuality', 'BoundaryFence', 'ExtraFacility', 'LoungeQuality']
    for col in none_cols:
        all_data[col] = all_data[col].fillna('None')

    for col in ['ZoningCategory', 'UtilityAccess', 'ElectricalSystem', 'KitchenQuality', 'PropertyFunctionality']:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # --- Handle Categorical Features (Ordinal & Nominal) ---
    
    # Ordinal (Ordered) Features: Map them to numbers
    quality_mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    ordinal_cols = ['ExteriorQuality', 'ExteriorCondition', 'BasementHeight', 'BasementCondition',
                    'HeatingQuality', 'KitchenQuality', 'LoungeQuality', 'ParkingQuality',
                    'ParkingCondition', 'PoolQuality']
    
    for col in ordinal_cols:
        all_data[col] = all_data[col].map(quality_mapping).fillna(0)
    
    all_data['BasementExposure'] = all_data['BasementExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}).fillna(0)
    all_data['PropertyFunctionality'] = all_data['PropertyFunctionality'].map({'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}).fillna(7)
    all_data['ParkingFinish'] = all_data['ParkingFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}).fillna(0)

    # Nominal (Unordered) Features: Convert to 'category' dtype for LightGBM
    object_cols = all_data.select_dtypes(include=['object']).columns
    for col in object_cols:
        all_data[col] = all_data[col].astype('category')
    
    print("Nominal features converted to 'category' dtype for LightGBM.")

    # --- Feature Engineering ---
    
    all_data['PropertyAge'] = all_data['YearSold'] - all_data['ConstructionYear']
    all_data['AgeSinceRemodel'] = all_data['YearSold'] - all_data['RenovationYear']
    all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
    all_data['TotalBaths'] = all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) + \
                           all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths'])
    all_data['TotalPorchSF'] = all_data['TerraceArea'] + all_data['OpenVerandaArea'] + \
                              all_data['EnclosedVerandaArea'] + all_data['SeasonalPorchArea'] + \
                              all_data['ScreenPorchArea']
    all_data['HasPool'] = all_data['SwimmingPoolArea'].apply(lambda x: 1 if x > 0 else 0)
    all_data['OverallQuality_Cond'] = all_data['OverallQuality'] * all_data['OverallCondition']
    
    print(f"Feature engineering complete. New data shape: {all_data.shape}\n")
    
    # Split data back into train and test
    X = all_data[:ntrain]
    X_test = all_data[ntrain:]
    
    return X, X_test, y_train, test_ids

def train_model(X, y_train, X_test):
    """Trains a Linear Regression model using K-Fold Cross-Validation."""
    print("--- 4. Model Training (Linear Regression only) ---")

    # Separate categorical and numerical features
    categorical_features = X.select_dtypes(include=['category']).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # K-Fold Cross-Validation setup
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # Arrays to store predictions
    oof_predictions = np.zeros(X.shape[0])
    test_predictions = np.zeros(X_test.shape[0])

    # Initialize OneHotEncoder
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    for fold, (train_index, val_index) in enumerate(kf.split(X, y_train)):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # One-hot encode categorical features
        X_train_ohe = ohe.fit_transform(X_train_fold[categorical_features])
        X_val_ohe = ohe.transform(X_val_fold[categorical_features])

        # Convert to DataFrame with string column names
        X_train_ohe = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names_out().astype(str))
        X_val_ohe = pd.DataFrame(X_val_ohe, columns=ohe.get_feature_names_out().astype(str))

        # Concatenate numerical features
        X_train_lin = pd.concat([X_train_ohe, X_train_fold[numerical_features].reset_index(drop=True)], axis=1)
        X_val_lin = pd.concat([X_val_ohe, X_val_fold[numerical_features].reset_index(drop=True)], axis=1)

        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train_lin, y_train_fold)

        # Validation predictions
        val_preds = model.predict(X_val_lin)
        oof_predictions[val_index] = val_preds

        # Test set predictions
        X_test_ohe = ohe.transform(X_test[categorical_features])
        X_test_ohe = pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names_out().astype(str))
        X_test_lin = pd.concat([X_test_ohe, X_test[numerical_features].reset_index(drop=True)], axis=1)
        test_predictions += model.predict(X_test_lin) / N_SPLITS

    # Calculate overall OOF RMSE
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    print(f"\nTraining complete.")
    print(f"Overall OOF RMSE (on log-transformed data): {oof_rmse:.5f}\n")

    return test_predictions

def create_submission(test_ids, test_predictions):
    """Creates the submission.csv file."""
    print("--- 5. Creating Submission ---")
    
    # **Inverse Transform Predictions**
    # We must convert the log-transformed predictions back to their original scale
    final_predictions = np.expm1(test_predictions)

    # Ensure no negative predictions
    final_predictions[final_predictions < 0] = 0

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'HotelValue': final_predictions
    })

    # Save to CSV
    submission_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Submission file created successfully at: {OUTPUT_FILE}")
    print(submission_df.head())


# --- Main execution ---
if __name__ == "__main__":
    print("x`Starting Hotel Value Prediction Pipeline...")
    
    # Step 1: Load
    train_df, test_df, sample_sub = load_data()
    
    if train_df is not None:
        # Step 2: EDA
        perform_eda(train_df)
        
        # Step 3: Preprocessing & Feature Engineering
        X, X_test, y_train, test_ids = preprocess_and_feature_engineer(train_df, test_df)
        
        # Step 4: Model Training
        test_predictions = train_model(X, y_train, X_test)
        
        # Step 5: Submission
        create_submission(test_ids, test_predictions)
        
        print("\n Pipeline finished successfully!")
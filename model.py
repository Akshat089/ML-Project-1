import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

# ---------------- Configuration ----------------
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
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
    for df in [X_train, X_test]:
        df['BasementFacilitySF1'] = df['BasementFacilitySF1'].fillna(0)
        df['BasementFacilitySF2'] = df['BasementFacilitySF2'].fillna(0)
        df['Type1_Score'] = df['BasementFacilityType1'].fillna('None').map(basement_map).fillna(0)
        df['Type2_Score'] = df['BasementFacilityType2'].fillna('None').map(basement_map).fillna(0)
        df['TotalBasementScore'] = df['Type1_Score']*df['BasementFacilitySF1'] + df['Type2_Score']*df['BasementFacilitySF2']
        df['BasementFinishedSF'] = df['BasementFacilitySF1'] + df['BasementFacilitySF2']
        df.drop(columns=['BasementFacilityType1','BasementFacilityType2','BasementFacilitySF1','BasementFacilitySF2','Type1_Score','Type2_Score'], errors='ignore', inplace=True)

    # ---------------- Pool Features ----------------
    pool_map = {'None':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}
    for df in [X_train, X_test]:
        df['SwimmingPoolArea'] = df['SwimmingPoolArea'].fillna(0)
        df['PoolQuality'] = df['PoolQuality'].fillna('None')
        df['TotalPoolScore'] = df['SwimmingPoolArea'] * df['PoolQuality'].map(pool_map).fillna(0)
        df.drop(columns=['PoolQuality','SwimmingPoolArea'], errors='ignore', inplace=True)

    # ---------------- Porch Features ----------------
    for df in [X_train, X_test]:
        df['TotalPorchArea'] = df[['OpenVerandaArea','EnclosedVerandaArea','SeasonalPorchArea','ScreenPorchArea']].fillna(0).sum(axis=1)
        df.drop(columns=['OpenVerandaArea','EnclosedVerandaArea','SeasonalPorchArea','ScreenPorchArea'], errors='ignore', inplace=True)

    # ---------------- Other Feature Engineering ----------------
    for df in [X_train, X_test]:
        # Parking
        quality_map_5pt = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}
        finish_map = {'Fin':3,'RFn':2,'Unf':1,'None':0}
        df['ParkingQuality'] = df['ParkingQuality'].fillna('None').map(quality_map_5pt).fillna(0)
        df['ParkingCondition'] = df['ParkingCondition'].fillna('None').map(quality_map_5pt).fillna(0)
        df['ParkingFinish'] = df['ParkingFinish'].fillna('None').map(finish_map).fillna(0)
        # House age
        df['HouseAge'] = df['YearSold'] - df['ConstructionYear']
        df['RenovationYear'] = df['RenovationYear'].fillna(df['ConstructionYear'])
        df.loc[df['RenovationYear']==0,'RenovationYear'] = df.loc[df['RenovationYear']==0,'ConstructionYear']
        df['YearsSinceModification'] = df['YearSold'] - df[['ConstructionYear','RenovationYear']].max(axis=1)
        # Interaction
        df['QualityArea'] = df['OverallQuality'] * df['UsableArea']
        df['TotalBathrooms'] = df['FullBaths'] + 0.5*df['HalfBaths']
        # Log skewed numeric features
        for col in ['RoadAccessLength','LandArea','FacadeArea','BasementTotalSF','ParkingArea']:
            if col in df.columns:
                df[col+'_Log'] = np.log1p(df[col].fillna(0))

    # Drop unused columns
    drop_cols = ['Id','BoundaryFence','ExtraFacility','ServiceLaneType','BasementHalfBaths','LowQualityArea','FacadeType','ParkingArea']
    X_train.drop(columns=drop_cols, errors='ignore', inplace=True)
    X_test.drop(columns=drop_cols, errors='ignore', inplace=True)

    print(f"Final training shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")
    return X_train, X_test, y_train, test_ids

# ----------------- 4. Model Training -----------------
def train_model(X, y, X_test):
    print("--- 4. Model Training: LassoCV ---")
    # Numeric/categorical separation
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('lasso', LassoCV(alphas=np.logspace(-5,1,100), cv=25, max_iter=10000, random_state=RANDOM_SEED))
    ])
    
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

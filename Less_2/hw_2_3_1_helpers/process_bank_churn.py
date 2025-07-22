import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Any, Optional

def select_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Selects input and target columns from the given DataFrame.
    """
    input_cols = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    target_col = 'Exited'
    X = df[input_cols]
    y = df[target_col]
    return X, y, input_cols

def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and validation sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_numeric(
    train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales numeric features using StandardScaler.
    """
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_scaled[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_scaled, val_scaled, scaler

def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the 'Gender' column as binary (Male=1, Female=0).
    """
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    return df

def encode_geography(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
    """
    One-hot encodes the 'Geography' column.
    """
    geo_train = pd.get_dummies(train_df['Geography'], prefix='Geography', drop_first=True)
    geo_val = pd.get_dummies(val_df['Geography'], prefix='Geography', drop_first=True)
    train_df = pd.concat([train_df.drop(columns='Geography'), geo_train], axis=1)
    val_df = pd.concat([val_df.drop(columns='Geography'), geo_val], axis=1)
    encoder = None  # Placeholder if you later want to implement a dedicated encoder
    return train_df, val_df, encoder

def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[StandardScaler], Optional[Any]]:
    """
    Full preprocessing pipeline for bank churn data.
    Returns:
        X_train: pd.DataFrame — features for training
        train_targets: pd.Series — targets for training
        X_val: pd.DataFrame — features for validation
        val_targets: pd.Series — targets for validation
        input_cols: List[str] — final list of feature columns (after preprocessing)
        scaler: Optional[StandardScaler] — fitted scaler (if used)
        encoder: Optional[Any] — encoder for categorical features (None by default)
    """
    X, y, input_cols = select_columns(raw_df)
    X_train, X_val, train_targets, val_targets = split_data(X, y)
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if scaler_numeric:
        X_train, X_val, scaler = scale_numeric(X_train, X_val, numeric_cols)
    else:
        scaler = None
    X_train = encode_gender(X_train)
    X_val = encode_gender(X_val)
    X_train, X_val, encoder = encode_geography(X_train, X_val)
    input_cols_final = X_train.columns.tolist()
    return X_train, train_targets, X_val, val_targets, input_cols_final, scaler, encoder

def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler] = None,
    reference_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Preprocesses new data (e.g., test set) using already fitted scaler and encoder.
    Automatically selects only those columns from new_df that are present in input_cols.
    Adds missing columns from reference_columns with zeros and reorders columns to match training set.
    Returns a DataFrame ready for model inference.
    """
    # Use only columns present in both input_cols and new_df.columns
    used_cols = [col for col in input_cols if col in new_df.columns]
    X_new = new_df[used_cols].copy()

    # Scale numeric columns if scaler is provided and columns exist
    if scaler is not None:
        numeric_cols = [col for col in X_new.select_dtypes(include=['float64', 'int64']).columns if col in X_new.columns]
        if numeric_cols:
            X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])

    # Encode gender if column exists
    if 'Gender' in X_new.columns:
        X_new = encode_gender(X_new)

    # One-hot encode geography if column exists
    if 'Geography' in X_new.columns:
        geo_new = pd.get_dummies(X_new['Geography'], prefix='Geography', drop_first=True)
        X_new = pd.concat([X_new.drop(columns='Geography'), geo_new], axis=1)

    # Add missing columns and reorder according to reference (usually train columns)
    if reference_columns is not None:
        for col in reference_columns:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[reference_columns]

    # Warn if columns are missing (optional)
    missing_cols = [col for col in input_cols if col not in new_df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing in new_df and will not be used: {missing_cols}")

    return X_new



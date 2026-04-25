import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

TARGET = 'burnout_level'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[load]    Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != TARGET]

    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    if df[TARGET].isnull().sum() > 0:
        df[TARGET] = df[TARGET].fillna(df[TARGET].mode()[0])

    print(f"[missing] Remaining missing values: {df.isnull().sum().sum()}")
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    le = LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    df[TARGET] = le.transform(df[TARGET])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"[encode]  Label mapping: {mapping}")
    return df, mapping


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print(f"[scale]   X: {X_scaled.shape}, y: {y.shape}")
    return X_scaled, y


def split_and_save(X: pd.DataFrame, y: pd.Series, output_dir: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df  = pd.concat([X_test.reset_index(drop=True),  y_test.reset_index(drop=True)],  axis=1)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'),   index=False)

    print(f"[split]   Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"[save]    Output: {output_dir}/")


def run_preprocessing(input_path: str, output_dir: str):
    df = load_data(input_path)
    df = handle_missing_values(df)
    df, _ = encode_target(df)
    X, y = scale_features(df)
    split_and_save(X, y, output_dir)
    print("\nPreprocessing Selesai.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated preprocessing untuk Developer Burnout dataset.')
    parser.add_argument('--input',  type=str, default='../developer_burnout_raw/developer_burnout.csv')
    parser.add_argument('--output', type=str, default='developer_burnout_preprocessing')
    args = parser.parse_args()

    run_preprocessing(args.input, args.output)

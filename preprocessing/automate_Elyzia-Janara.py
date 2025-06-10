import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import os

def load_and_preprocess(csv_path, output_path=None):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Drop missing values (jika ada)
    df.dropna(inplace=True)

    # Drop duplikat
    df.drop_duplicates(inplace=True)

    # Label Encoding
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    df['Product_importance'] = df['Product_importance'].map({'low': 0, 'medium': 1, 'high': 2})

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Warehouse_block', 'Mode_of_Shipment'], drop_first=True)

    # Drop kolom ID (tidak diperlukan untuk model)
    df.drop(columns=['ID'], inplace=True)

    # Fitur numerik untuk standarisasi & deteksi outlier
    numeric_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
                    'Prior_purchases', 'Discount_offered', 'Weight_in_gms']

    # Deteksi & hapus outlier dengan Z-score
    z_scores = np.abs(zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]

    # Normalisasi (Standard Scaler)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Split fitur dan label
    X = df.drop('Reached.on.Time_Y.N', axis=1)
    y = df['Reached.on.Time_Y.N']

    # Split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save preprocessed dataset (opsional)
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        X_train.to_csv(os.path.join(output_path, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_path, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_path, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_path, 'y_test.csv'), index=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Contoh penggunaan
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(repo_root, 'ecommerce_shipping_data', 'ecommerce_shipping_data.csv')
    output_dir = "ecommerce_shipping_data_preprocessed"
    load_and_preprocess(input_csv, output_dir)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import logging
import os
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessor with configuration parameters.   
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataframe.
        """
        initial_size = len(df)
        df_cleaned = df.drop_duplicates()
        final_size = len(df_cleaned)

        logger.info(f"Original dataset size: {initial_size:,} rows")
        logger.info(f"After removing duplicates: {final_size:,} rows")

        return df_cleaned

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_column: str = "Class",
        columns_to_drop: list = ["Time"],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features by dropping specified columns and separating features from target.
        """
        # Drop unnecessary columns
        for column in columns_to_drop:
            if column in df.columns:
                df = df.drop(column, axis=1)

        # Separate features and target
        X = df.drop([target_column], axis=1).astype("float32")
        y = df[target_column]

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {dict(y.value_counts())}")

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_ratio: float = 0.5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        """
        # First split: separate train from temp (which will become val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Second split: separate validation from test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_temp,
        )

        # Log split information
        total_samples = len(X)
        logger.info("Data split completed:")
        logger.info(
            f"   Training set: {X_train.shape[0]:,} samples "
            f"({X_train.shape[0]/total_samples*100:.1f}%)"
        )
        logger.info(
            f"   Validation set: {X_val.shape[0]:,} samples "
            f"({X_val.shape[0]/total_samples*100:.1f}%)"
        )
        logger.info(
            f"   Test set: {X_test.shape[0]:,} samples "
            f"({X_test.shape[0]/total_samples*100:.1f}%)"
        )

        # Verify fraud distribution
        for name, target in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
            fraud_pct = target.mean() * 100
            logger.info(f"   {name} fraud rate: {fraud_pct:.3f}%")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    def scale_features(self, split_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        """
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(split_data["X_train"]),
            columns=split_data["X_train"].columns,
            index=split_data["X_train"].index,
        )

        # Transform validation and test sets
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(split_data["X_val"]),
            columns=split_data["X_val"].columns,
            index=split_data["X_val"].index,
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(split_data["X_test"]),
            columns=split_data["X_test"].columns,
            index=split_data["X_test"].index,
        )

        logger.info("Scaled features statistics:")
        logger.info(f"   Training mean: {X_train_scaled.mean().mean():.6f}")
        logger.info(f"   Training std: {X_train_scaled.std().mean():.6f}")

        return {
            "X_train": X_train_scaled,
            "y_train": split_data["y_train"],
            "X_val": X_val_scaled,
            "y_val": split_data["y_val"],
            "X_test": X_test_scaled,
            "y_test": split_data["y_test"],
        }


def save_datasets(data: Dict[str, pd.DataFrame], output_dir: str, scaled: bool):
    """
    Save processed datasets to specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save each dataset
    for prefix in ["train", "val", "test"]:
        X = data[f"X_{prefix}"]
        y = data[f"y_{prefix}"]

        # Restructure as label first, then features
        full_data = pd.concat([y, X], axis=1)

        # Save to CSV
        if not scaled:
            filename = f"{prefix}_raw_data.csv"
        else:
            filename = f"{prefix}_data.csv"
        
        output_path = os.path.join(output_dir, filename)
        full_data.to_csv(output_path, index=False, header=False)
        logger.info(f"Saved {prefix} data to {output_path}")


def main():
    """Main processing pipeline."""
    try:
        # Load data
        input_data_path = os.path.join("/opt/ml/processing/input", "creditcard.csv")
        logger.info(input_data_path)
        df = pd.read_csv(input_data_path)

        # Initialize preprocessor
        preprocessor = DataPreprocessor()

        # Process data
        df_cleaned = preprocessor.remove_duplicates(df)
        X, y = preprocessor.prepare_features(df_cleaned)
        split_data = preprocessor.split_data(X, y)
        scaled_data = preprocessor.scale_features(split_data)

        # Save processed datasets
        output_dir = "/opt/ml/processing/output"
        save_datasets(split_data, output_dir, False)  # Raw data
        save_datasets(scaled_data, output_dir, True)  # Scaled data

        logger.info("Preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
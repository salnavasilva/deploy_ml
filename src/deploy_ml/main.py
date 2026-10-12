
from deploy_ml.data.load_data import CatsDogsLoader
from deploy_ml.model.model import ImageModel
from deploy_ml.preprocessing.data_preprocessor import DataPreprocess
from deploy_ml.preprocessing.transpose_df_cols import DataFrameTransformer
import joblib
import os

def main():
    loader = CatsDogsLoader(split="train", streaming=False)
    preprocessor = DataPreprocess(loader, batch_size=64, max_batches=200).image_embedder_batch()
    transformer = DataFrameTransformer(preprocessor).transform_rows2df()
    
    model = ImageModel(transformer).train_model()

    print(f"Training rows: {len(transformer)}")
    print(f"Accuracy: {model.accuracy:.4f}")

    # Save the model artifact
    model_path = "model.joblib"
    joblib.dump(model.model, model_path)
    print(f"Model saved to {model_path}")

    # Test data 
    test_loader = CatsDogsLoader(split="validation", streaming=False)
    test_preprocessor = DataPreprocess(test_loader, batch_size=64, max_batches=10).image_embedder_batch()
    test_transformer = DataFrameTransformer(test_preprocessor).transform_rows2df()
    y_pred = ImageModel(test_transformer).predict()
    print(f"Test predictions: {y_pred}")

if __name__ == "__main__":
    main()

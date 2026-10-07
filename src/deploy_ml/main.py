
from deploy_ml.data.load_data import CatsDogsLoader
from deploy_ml.preprocessing.data_preprocessor import DataPreprocess
from deploy_ml.preprocessing.transpose_df_cols import DataFrameTransformer


def main():
    loader = CatsDogsLoader(split="train", streaming=False)
    preprocessor = DataPreprocess(loader, batch_size=2, max_batches=2).image_embedder_batch()
    transformer = DataFrameTransformer(preprocessor).transform_rows2df()
    print(transformer)
    print(type(transformer))
    

    

    # for j, (image, label) in enumerate(preprocessor.image_embedder_batch(batch_size=2, max_batches=2)):
    #     print(image, label)

    #     if j == 2:
    #         break

if __name__ == "__main__":
    main()

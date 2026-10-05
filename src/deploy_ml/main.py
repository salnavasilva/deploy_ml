from deploy_ml.data.load_data import CatsDogsLoader
from deploy_ml.preprocessing.data_preprocessor import DataPreprocess


def main():
    loader = CatsDogsLoader(split="train")
    preprocessor = DataPreprocess(loader, batch_size=2, max_batches=2).image_embedder_batch()
    for embeddings, labels in preprocessor:
        print(embeddings)
        print(labels)
    

    # for j, (image, label) in enumerate(preprocessor.image_embedder_batch(batch_size=2, max_batches=2)):
    #     print(image, label)

    #     if j == 2:
    #         break

if __name__ == "__main__":
    main()

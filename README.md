
# Project Briefing: Cat vs. Dog Image Classifier

## Project Objective
The primary objective of this project is to build and deploy a machine learning pipeline that can distinguish between images of cats and dogs. The goal is to create a robust end-to-end flow: from loading raw image data to preprocessing it into embeddings, training a classification model, and evaluating its performance.

## Business Context
This project serves as a foundational implementation for image classification tasks. By leveraging pre-trained embeddings (via a CLIP-like encoder), the system aims to achieve high accuracy with relatively low computational overhead compared to training a deep neural network from scratch. This approach is suitable for rapid prototyping and deployment in scenarios where high-level semantic understanding of images is required.

## Available Data
- **Dataset**: `microsoft/cats_vs_dogs` from the Hugging Face `datasets` library.
- **Data Structure**: The dataset contains images and corresponding labels (cat/dog).
- **Processing**: 
    - Images are converted to RGB.
    - Labels are extracted as numpy arrays.
    - Embeddings are generated using a `ClipEncoder` (from the `embetter` library).
    - Data is transformed into a pandas DataFrame for easier manipulation and model training.

## Assumptions
- **Embedding Quality**: It is assumed that the `ClipEncoder` provides sufficiently discriminative embeddings for the cat vs. dog classification task.
- **Data Distribution**: The training and validation splits are assumed to be representative of the overall data distribution.
- **Hardware**: The current implementation assumes sufficient memory to hold the processed embeddings in a pandas DataFrame.
- **Library Availability**: The project assumes the presence of `datasets`, `numpy`, `pandas`, `scikit-learn`, and `embetter` libraries.

## Decisions Taken
- **Model Choice**: `LogisticRegression` from `scikit-learn` is used as the primary classifier, chosen for its efficiency and effectiveness on high-dimensional embeddings.
- **Preprocessing Strategy**: Instead of raw pixel values, the pipeline uses image embeddings to reduce dimensionality and improve generalization.
- **Data Handling**: A custom `DataFrameTransformer` is used to explode the batch-processed embeddings into a flat DataFrame structure suitable for `scikit-learn`'s `train_test_split`.
- **Training Split**: A default `test_size` of 0.2 with a `random_state` of 42 is used for reproducibility.

## Experiments Completed
- **Batch Processing**: Implemented a batch-based loading and preprocessing mechanism (`DataPreprocess`) to handle large datasets efficiently.
- **Stratified Splitting**: Added logic to handle stratified splitting in `ImageModel` to ensure balanced class distribution in training and test sets, with fallback logic for small datasets.
- **Streaming Support**: The `CatsDogsLoader` includes a `streaming` parameter, though it is currently set to `False` for the standard training flow.

## Open Questions
- **Scalability**: How will the pipeline perform if the number of images grows significantly? (Currently, all embeddings are loaded into memory in a DataFrame).
- **Encoder Selection**: Should we explore different encoders (e.g., ViT, ResNet) to see if they provide better accuracy for this specific task?
- **Deployment**: What is the target deployment environment (e.g., FastAPI, Streamlit, or a cloud function)?

## Next Steps
1. **Performance Evaluation**: Run the `main.py` script and record the baseline accuracy.
2. **Memory Optimization**: Investigate using `dask` or a generator-based approach for the `DataFrameTransformer` if memory limits are hit.
3. **Web Interface**: Create a simple UI (e.g., using Streamlit or a basic HTML/JS frontend) to allow users to upload an image and receive a prediction.
4. **Model Persistence**: Implement a method to save and load the trained `LogisticRegression` model using `joblib` or `pickle`.
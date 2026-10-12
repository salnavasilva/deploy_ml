import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ImageModel:

    def __init__(self, processed_data, test_size=0.2, random_state=42):
        self.processed_data = processed_data
        self.test_size = test_size
        self.random_state = random_state
        self.model = LogisticRegression(max_iter=1000)
        self.accuracy = None

    def _split_features_labels(self):
        X = np.array(self.processed_data["embeddings"].tolist(), dtype=float)
        y = np.array(self.processed_data["labels"].tolist())
        return X, y

    def train_model(self):
        X, y = self._split_features_labels()

        classes, counts = np.unique(y, return_counts=True)
        n_test = int(np.ceil(self.test_size * len(y)))
        stratify = (
            y
            if len(classes) > 1 and counts.min() >= 2 and n_test >= len(classes)
            else None
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        self.model.fit(X_train, y_train)
        self.accuracy = accuracy_score(y_test, self.model.predict(X_test))

        return self

    def predict(self, embeddings):
        X = np.asarray(embeddings, dtype=float)
        return self.model.predict(X)

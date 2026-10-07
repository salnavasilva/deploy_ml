# from embetter.vision import ImageLoader
import numpy as np
from embetter.multi import ClipEncoder


class DataPreprocess:

    def __init__(self, loader, batch_size=64, max_batches=None):
        self.loader = loader
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.encoder = ClipEncoder()

    def image_embedder_batch(self):

        for batch in self.loader.iter_batches(self.batch_size, self.max_batches):
            images = [img["image"].convert("RGB") for img in batch]
            labels = np.array([label["labels"] for label in batch])


            yield self.encoder.transform(images), labels


        


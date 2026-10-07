import pandas as pd


class DataFrameTransformer:
   def __init__(self, preprocessor):
       self.preprocessor = preprocessor
   
   def transform_rows2df(self) -> pd.DataFrame:
        rows = []
        for embeddings, labels in self.preprocessor:
            rows.append({"embeddings": embeddings, "labels": labels})

        return pd.DataFrame(rows).explode(['embeddings', 'labels']).reset_index(drop=True)
               
               

       
from datasets import load_dataset


class CatsDogsLoader:

    def __init__(self, split, streaming):
        self.dataset = load_dataset(
            "microsoft/cats_vs_dogs",
            split=split,
            streaming=streaming
        )

    def __iter__(self):
        return iter(self.dataset)
    

    def iter_batches(self, batch_size, max_batches):
        batch = []
        batch_num = 0

        for sample in self:
            batch.append(sample)

            if len(batch) == batch_size:
                yield batch
                batch = []

                batch_num += 1

            if max_batches is not None and batch_num >= max_batches:
                break

        if batch and (max_batches is None or batch_num < max_batches):
            yield batch




if __name__ == "__main__":
    loader = CatsDogsLoader(split="train", streaming=False)
    print(type(loader))

    for batch in loader.iter_batches(batch_size=5, max_batches=2):
        print(batch)
    
    # print 10 examples 
    # for i, sample in enumerate(loader):
    #     print(sample)

    #     if i == 9:
    #         break
        
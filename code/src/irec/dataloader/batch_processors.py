import torch
import itertools
from irec.utils import MetaParent


class BaseBatchProcessor(metaclass=MetaParent):
    def __call__(self, batch):
        raise NotImplementedError


class IdentityBatchProcessor(BaseBatchProcessor, config_name="identity"):
    def __call__(self, batch):
        return torch.tensor(batch)


class BasicBatchProcessor(BaseBatchProcessor, config_name="basic"):
    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith(".ids"):
                prefix = key.split(".")[0]
                length_key = f"{prefix}.length"
                assert length_key in batch[0]

                ids_iter = itertools.chain.from_iterable(s[key] for s in batch)
                processed_batch[key] = torch.tensor(list(ids_iter), dtype=torch.long)

                lengths_list = [s[length_key] for s in batch]
                processed_batch[length_key] = torch.tensor(
                    lengths_list, dtype=torch.long
                )

        for part, values in processed_batch.items():
            if not isinstance(processed_batch[part], torch.Tensor):
                processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch

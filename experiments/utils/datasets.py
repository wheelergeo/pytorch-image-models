import pyarrow.parquet as pq
import io


from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable


class ParquetImageDataset(Dataset):
    def __init__(self, parquet_files: list, transform: Optional[Callable] = None):
        self.samples = []
        for file in parquet_files:
            table = pq.read_table(file).flatten()
            columns = table.to_pydict()
            self.samples.extend(zip(columns['image.bytes'], columns['label']))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_bytes, label = self.samples[idx]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
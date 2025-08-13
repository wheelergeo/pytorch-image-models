import pyarrow.parquet as pq
import io
import os


from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable


class ParquetImageDataset(Dataset):
    def __init__(self, parquet_dir: str, transform: Optional[Callable] = None):
        assert os.path.exists(parquet_dir), f"Parquet directory {parquet_dir} does not exist"

        self.samples = []
        for fname in os.listdir(parquet_dir):
            table = pq.read_table(os.path.join(parquet_dir, fname)).flatten()
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
    

if __name__ == '__main__':
    from timm.data.dataset import ImageDataset
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    d = ImageDataset(root="/nfs5/yrc/dataset/imagenet-1k/ILSVRC2012_img_train.tar")
    
    print(len(d))
    
    l = DataLoader(d, batch_size=128, shuffle=True, num_workers=4)
    for img, label in l:
        print(img.shape, label)
        break
    
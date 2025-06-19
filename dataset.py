import torch
import torch.utils.data as data
from PIL import Image
import os

class DogCatDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        if split not in ('train', 'eval'):
            raise RuntimeError('Please specify loading the train/eval set!')
        self.transform = transform
        self.image_path = []
        self.tags = []

        path = os.path.join(root, split)
        cats = os.listdir(os.path.join(path, 'Cat'))
        cats.sort(key=lambda x: int(x.split('.')[0]))
        for file_path in cats:
            file_path = os.path.join(path, 'Cat', file_path)
            self.image_path.append(file_path)
            self.tags.append(torch.tensor([1.0])) # Cat is 1

        dogs = os.listdir(os.path.join(path, 'Dog'))
        dogs.sort(key=lambda x: int(x.split('.')[0]))
        for file_path in dogs:
            file_path = os.path.join(path, 'Dog', file_path)
            self.image_path.append(file_path)
            self.tags.append(torch.tensor([0.0])) # Dog is 1

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.tags[index]

    def __len__(self):
        return len(self.image_path)


if __name__ == '__main__':
    d = DogCatDataset('/Users/soda/Desktop/Cat&Dog')
    print(d[0])

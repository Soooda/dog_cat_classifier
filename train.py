import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os
import time

from model import CNN
from dataset import DogCatDataset

torch.manual_seed(990919)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_root = '/Users/soda/Desktop/Cat&Dog/'
checkpoint_path = 'checkpoints'
num_epochs = 100
batch_size = 128
learning_rate = 0.0004


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_dataset = DogCatDataset(root=data_root, split='train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
eval_dataset = DogCatDataset(root=data_root, split='eval', transform=transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model = CNN(224, 224).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=4, min_lr=0.00001)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Check for the latest checkpoint
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
    if checkpoints:
        latest_checkpoint = max([int(os.path.splitext(f)[0]) for f in checkpoints])
        checkpoint_file = os.path.join(checkpoint_path, f"{latest_checkpoint}.pth")
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = latest_checkpoint
            print(f'Resuming from epoch {start_epoch}')

for epoch in range(start_epoch + 1, num_epochs + 1):
    start = time.time()

    model.train()
    for image, tag in train_dataloader:
        image = image.to(device)
        tag = tag.to(device)

        pred = model(image)
        loss = criterion(pred, tag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    eval_loss = 0.0
    for image, tag in eval_dataloader:
        image = image.to(device)
        tag = tag.to(device)

        pred = model(image)
        loss = criterion(pred, tag)
        eval_loss += loss.item()
    scheduler.step(eval_loss)
    end = time.time()

    print(f"Epoch {epoch:<4}/{num_epochs} Eval Loss: {eval_loss:<8.4f} Time: {(end - start) / 60:.2f} min")

    with open('train.log', 'a') as f:
        f.write(f'Epoch {epoch:<4}/{num_epochs} | Train Loss: {eval_loss:<8.4f} | Time: {(end - start) / 60:.2f} min\n')

    checkpoints = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoints, os.path.join(checkpoint_path, f'{epoch}.pth'))

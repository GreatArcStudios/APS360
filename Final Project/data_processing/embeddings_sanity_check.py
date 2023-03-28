import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

embeddings_val_path = './embeddings/embeddingval/'

# Create a dataset for the embeddings in the validation set
embedding_val_dataset = DatasetFolder(embeddings_val_path, loader=torch.load, extensions=('.tensor'))

# Create a data loader for the embeddings in the validation set
embedding_val_loader = DataLoader(embedding_val_dataset, batch_size=1, shuffle=True)

def get_mean_std(loader, num_samples=10000):
    mean = 0
    std = 0
    processed_samples = 0
    
    for i, (images, _) in enumerate(loader):
        if i >= num_samples:
            break
        mean += torch.mean(images, dim=[2, 3])
        std += torch.std(images, dim=[2, 3], unbiased=False)
        processed_samples += 1

    mean /= processed_samples
    std /= processed_samples

    return mean, std

mean, std = get_mean_std(embedding_val_loader)
print(f"Mean: {mean}, Standard deviation: {std}, Shapes of mean and std: {mean.shape}, {std.shape}")

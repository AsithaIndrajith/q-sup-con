import os
import torch as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_dataset( input_shape, batch_size, data_dir, data_transforms ):
    image_datasets = {
    x if x == "train" else "val": datasets.ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]
    )
    for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    y_train = image_datasets["train"].targets
    y_test = image_datasets["val"].targets

    # Initialize dataloader
    dataloaders = {
    x: nn.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ["train", "val"]
    }

    train_dataset = image_datasets["train"]
    test_dataset = image_datasets["val"]

    train_loader = dataloaders['train']
    test_loader = dataloaders['val']

    print("Training dataset size: " + str(len(train_dataset)))
    print("Testing datset size: " + str(len(test_dataset)))

    return train_dataset, test_dataset, train_loader, test_loader, dataset_sizes, class_names, y_train, y_test

def get_stats(train_loader, input_shape):
    # https://kozodoi.me/blog/20210308/compute-image-stats
    psum    = nn.tensor([0.0, 0.0, 0.0])
    psum_sq = nn.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs, labels in train_loader:
        psum    += inputs.sum(axis        = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    ####### FINAL CALCULATIONS

    # pixel count
    count = len(train_loader.dataset) * input_shape[0] * input_shape[0]

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = nn.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

    return total_mean, total_std
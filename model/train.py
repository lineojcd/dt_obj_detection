import torch
import torchvision
import os
import tqdm
from dataset import DuckieSimDataset
import json

# local import
from pytorch_detection import utils
from pytorch_detection import transforms as T
from pytorch_detection.engine import train_one_epoch, evaluate

from model import Model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of `./weights`!
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load splist, create pytorch dataset
    # with open("./model/data_splits/new_splits.json", "r") as f:
    # with open("./model/data_splits/new_splits.json", "r") as f:
    with open("./model/data_splits/default_splits.json", "r") as f:
        splits = json.load(f)
    dataset = DuckieSimDataset(
        "./data_collection/dataset",
        splits,
        train=True,
        transforms=get_transform(train=True))
    dataset_val = DuckieSimDataset(
        "./data_collection/dataset",
        splits,
        train=False,
        transforms=get_transform(train=False))

    # # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_val = torch.utils.data.Subset(dataset_val, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model
    num_classes = 5
    model = Model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    print("training initialization finished...")
    print("DO NOT forget to dump output to logfile!")

    # let's train it for 10 epochs
    num_epochs = 9
    save_ckpt_step = 1
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the val dataset
        evaluate(model, data_loader_val, device=device)
        if epoch % save_ckpt_step == 0:
            name = f"new_faster_rcnn_resnet50_{'sgd'}_epoch{epoch}.ckpt"
            torch.save(model.state_dict(),
                os.path.join("./model/weights", name))

    print("Model finished")
    pass


if __name__ == "__main__":
    cwd = os.getcwd()
    cwd = cwd.partition('model')[0]
    # print(sta.partition(stb)[0])
    print(cwd)
    os.chdir(cwd)
    main()
import sys 
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import random
import json
import numpy as np 
import time

from torch import optim

from src.models.helpers import save_checkpoint, load_checkpoint, plot_losses, train, test
from src.models.models.cbam_cifar10 import ResNetk

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

def main():
    args = sys.argv
    torch.autograd.set_detect_anomaly(True)

    # load parameters
    project_parameters = json.load(open(args[1]))
    training_parameters = json.load(open(args[2]))
    data_path = args[3]

    # set random seed and device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    seed = project_parameters["random_seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


    # datasets and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(data_path, train=True, transform= transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(data_path, train=False, transform= transform, download=True)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = training_parameters["batch_size"], shuffle= True, num_workers= training_parameters["num_workers"])
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = training_parameters["batch_size"], shuffle= False, num_workers= training_parameters["num_workers"])


    
    # loading model or restart former training
    if training_parameters["load_model"] != "":
        net, optimizer, training_parameters, epoch_start = load_checkpoint(training_parameters["load_model"])
    else:
        net = ResNetk(
            k = training_parameters["resnet_depth"],
            reduction_ratio=training_parameters["reduction_ratio"], 
            kernel_cbam = training_parameters["kernel_cbam"],
            use_cbam_block= training_parameters["use_cbam_block"],
            use_cbam_class= training_parameters["use_cbam_class"]

            )
        epoch_start = 0
        optimizer = optim.SGD(net.parameters(),lr = training_parameters["lr"], momentum= training_parameters["momentum"])

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()


    # legends for plots
    legends = ["Training BCE(epoch)", "Training BCE(batch)", "Test accuracies"]
    xlabels = ["nb epochs", "nb_iterations", "nb_epochs"]
    ylabels = ["BCE loss", "BCE loss", "Accuracy"]

    # losses 
    epoch_losses = []
    iter_losses = []
    accuracies = []
    best_acc = 0

    # Start training
    s = time.time()
    for epoch in range(epoch_start, training_parameters["n_epochs"]):
        batch_losses = train(trainloader, optimizer, net, criterion, training_parameters, epoch, device)
        accuracy = test(testloader, net, device)

        iter_losses += batch_losses
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        accuracies.append(accuracy)

        print("Epoch n°{}, bce loss: {}, test accuracy: {}, duration {}".format(epoch,epoch_loss, accuracy, time.time() - s))

        # plot losses
        plot_losses(
            [epoch_losses, iter_losses, accuracies],
             legends, xlabels, ylabels, 
             project_parameters["losses"], 
             training_parameters["model_name"], 
             epoch)

        # save best model
        if accuracy > best_acc:
            best_acc = accuracy 
            save_checkpoint(net, optimizer, training_parameters, project_parameters["models"], epoch, best = True )

        # save model every save_every epoch
        if epoch % training_parameters["save_every"] == 0:
            save_checkpoint(net, optimizer, training_parameters, project_parameters["models"], epoch )




    


    


    
   




if __name__ == "__main__":
    main()
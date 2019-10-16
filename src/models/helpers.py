import matplotlib.pyplot as plt
import torch

def save_checkpoint(net, optimizer, training_parameters, path, epoch, best = False):
    checkpoint = {'model': net,
            'state_dict': net.state_dict(),
            'optimizer_state' : optimizer.state_dict(),
            'optimizer' : optimizer,
            'training_parameters' : training_parameters,
            'epoch': epoch + 1
            }

    if best:
        save_path = "{}best_{}.pth".format(path, training_parameters["model_name"])
    else:
        save_path = "{}{}_{}.pth".format(path, epoch, training_parameters["model_name"])


    torch.save(checkpoint, save_path)
    print("Model saved in {}".format(save_path))
    return save_path


def load_checkpoint(filepath, evaluation = False):
    print("Loading model from {}".format(filepath))
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']

    training_parameters = checkpoint["training_parameters"]
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint["epoch"]


    if evaluation:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return model, training_parameters
    
    return model, optimizer, training_parameters, epoch
    


def plot_losses(losses, legends, xlabels, ylabels, path, name, epoch ):
    nb_plots = len(losses)

    fig, axs = plt.subplots(nb_plots,1, squeeze=False)
    for ax, loss, label, xlabel, ylabel in zip(axs.flat, losses, legends, xlabels, ylabels):
        ax.plot(loss, label = label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend()
    # fig.tight_layout()
    plt.savefig("{}{}_losses_epoch_{}.jpg".format(path,name,epoch))
    plt.close()


def train(trainloader, optimizer, net, criterion, training_parameters, epoch, device):
    batch_losses = []
    for batch,data in enumerate(trainloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

        if batch % training_parameters["batch_every"] == 0:
            print("--epoch{}, batch n°{}, batch_bce_loss: {}".format(epoch, batch, loss.item()))
    return batch_losses

def test(testloader, net, device):
    correct = 0
    total = 0
    for batch,data in enumerate(testloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = net(imgs)

        maxs = torch.argmax(outputs, dim = 1)
        corrects = (maxs == labels).cpu().numpy().astype(int)
        correct += corrects.sum()
        total += imgs.size()[0]
    accuracy = correct/total
    return accuracy
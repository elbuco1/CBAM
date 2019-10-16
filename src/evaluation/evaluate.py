import sys 
import json
import torch
import torchvision
import random
import torchvision.transforms as transforms

from src.models.helpers import load_checkpoint, test

def main():

    print("Evaluation")
    args = sys.argv

    # load parameters
    project_parameters = json.load(open(args[1]))
    evaluation_parameters = json.load(open(args[2]))
    data_path = args[3]

    # set random seed and device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = project_parameters["random_seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


    # datasets and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    test_dataset = torchvision.datasets.CIFAR10(data_path, train=False, transform= transform, download=True)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = evaluation_parameters["batch_size"], shuffle= False, num_workers= evaluation_parameters["num_workers"])

    model, training_parameters = load_checkpoint(evaluation_parameters["model_path"], evaluation= True)

    model = model.to(device)

    accuracy = test(testloader, model, device)

    print("Evaluation accuracy: {}%".format(accuracy*100))

if __name__ == "__main__":
    main()
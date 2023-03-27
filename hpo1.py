#TODO: Import your dependencies.
import os
import logging
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader,hook,device, criterion):
    model.eval()
    test_loss = 0
    run_loss=0
    hook.set_mode(smd.modes.EVAL)
    total_corrects = 0
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for (data, total_outputs) in test_loader:
        outputs = model(data)
        test_loss += criterion(outputs, total_outputs).item()
        preds = torch.max(outputs, 1)
        total_corrects += torch.sum(preds==labels.data).item()
        average_accuracy = running_corrects / len(test_loader.dataset)
        average_loss = test_loss / len(test_loader.dataset)
        logger.info(f"Test set: average loss: {average_loss}, Average accuracy: {100*average_accuracy}%")
    pass

def train(model,train_loader,criterion, optimizer,validate_loader,epochs,hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    count = 0
    for epoch in range(epochs):
        print(epoch)
        
        # training
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        for (data, total_outputs) in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, total_outputs)
            loss.backward()
            optimizer.step()
            count += len(inputs)
        
        # validaiton
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        total_corrects = 0
        with torch.no_grad():
            for (data, total_outputs) in validation_loader:
                outputs = model(data)
                loss = criterion(data, total_outputs)
                preds = torch.max(outputs, 1)
                total_corrects += torch.sum(preds == total_outputs.data).item()
        total_accuracy = total_corrects / len(validation_loader.dataset)
        logger.info(f"Validation set: Average accuracy: {100*total_accuracy}%")
    
    return model    
    pass
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet40(pretrained=True)

    for param in model.parameters():
        param.require_grad = False
    
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133) ##Because of there are 133 classes of dog breeds, output of our model should be 133
                            )
    pass

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model,train_loader,criterion,optimizer,validate_loader,epochs,hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, hook,device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth")))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
     parser.add_argument(
        "--batch-size",
        type = int ,
        default = 64, 
        metavar = "N",
        help = "input batch size for training (default : 64)"
    )
    parser.add_argument(
        "--test-batch-size",
        type = int ,
        default = 1000, 
        metavar = "N",
        help = "input test batch size for training (default : 1000)"
    )
    parser.add_argument(
        "--epochs",
        type = int ,
        default = 7, 
        metavar = "N",
        help = "number of epochs to train (default : 7)"
    )
    parser.add_argument(
        "--lr",
        type = float ,
        default = 0.001, 
        metavar = "LR",
        help = "learning rate (default : 0.001)"
    )
    parser.add_argument("--hosts", type=list,
                        default=json.loads(os.environ["model_hosts"]))
    parser.add_argument("--current-host", type=str,
                        default=os.environ["model_current_hosts"])
    parser.add_argument("--model-dir", type=str,
                        default=os.environ["model_dir"])
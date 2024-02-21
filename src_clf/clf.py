import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import argparse
from datetime import datetime
from tqdm import tqdm

from transforms import *
from model import *
from utils.funcs import *
from utils.logger import *

import warnings 
warnings.filterwarnings('ignore') 

def main(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('mps')

    now=datetime.now()
    save_path='output/{}_'.format(args.session_name)+now.strftime("%m_%d_%H_%M")+'/'
    # save_path = '/Users/gufran/Developer/Projects/AI/CancerDM/output/resnet101_01_12_07_55/'
    os.mkdir(save_path)
    os.mkdir(save_path+'weights/')
    os.mkdir(save_path+'logs/')
    os.mkdir(save_path+'graphs/')

    train_data_path = 'data/original/train'
    val_data_path = 'data/original/val'
    test_data_path = 'data/original/test'
    config_path = 'src_clf/configs/config.json'

    cfg=load_json(config_path)

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=get_transform(args.model_name, cfg['image_size']))
    val_dataset = datasets.ImageFolder(root=val_data_path, transform=get_transform(args.model_name, cfg['image_size']))
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=get_transform(args.model_name, cfg['image_size']))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    print("Train images:", len(train_dataloader)*cfg['batch_size'])
    print("Val images:", len(val_dataloader)*cfg['batch_size'], end="\n\n")

    model = get_model(args.model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    n_epoch = cfg['epoch']

    best_val_loss = float('inf')
    train_accs = []
    train_losses = []
    valid_accs = []
    valid_losses = []

    for epoch in range(n_epoch):
        #TRAINING
        model.train()
        train_loss, valid_loss, train_acc, valid_acc = 0.0,0.0,0.0,0.0
        for inputs, labels in tqdm(train_dataloader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels) #loss(outputs.logits if is_inception else outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()
            _, predicted = torch.max(outputs, 1) #torch.max(outputs.logits if is_inception else outputs, 1)
            train_acc += (predicted == labels).sum().item()

        train_acc = train_acc / (len(train_dataloader.dataset))
        train_loss = train_loss/len(train_dataloader)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        #VALIDATION
        model.eval()
        valid_loss, valid_acc = 0.0,0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                valid_acc += (predicted == labels).sum().item()

        valid_acc = valid_acc / (len(val_dataloader.dataset))
        valid_loss = valid_loss/len(val_dataloader)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)

        print(f'Epoch [{epoch+1}/{n_epoch}] - train loss: {train_loss:.4f}, train acc: {train_acc:.4f}', end=", ")
        print(f'val loss: {valid_loss:.4f}, val acc: {valid_acc:.4f}', end="\n\n")

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            save_model_path=os.path.join(save_path+'weights/',"model.tar")
            torch.save({"model":model.state_dict(),"epoch":epoch},save_model_path)

        log_train("", epoch+1, train_loss, train_acc, valid_loss, valid_acc, save_path+'logs/train.log')

    #TESTING
    torch_load=torch.load(save_path+'weights/model.tar')["model"]
    model.load_state_dict(torch_load)

    model.eval()
    test_loss, test_acc = 0.0,0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_acc += (predicted == labels).sum().item()

    test_acc = test_acc / (len(test_dataloader.dataset))
    test_loss = test_loss/len(test_dataloader)
    log_test(args.model_name, test_loss, test_acc, save_path+'logs/test.log')

    print(f"\nTesting - acc: {test_acc}, loss: {test_loss}")


    #SAVE GRAPHS
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'graphs', 'loss_graph.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train')
    plt.plot(valid_accs, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'graphs', 'acc_graph.png'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',dest='session_name')
    parser.add_argument('-m',dest='model_name')
    args=parser.parse_args()
    main(args)
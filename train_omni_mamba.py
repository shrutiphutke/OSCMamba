import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import math
from OmniMamba_v7 import VSSM as medmamba # import model
# import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator
from ptflops import get_model_complexity_info

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # data_flag = 'breastmnist'
    # data_flag = 'octmnist'
    # data_flag = 'bloodmnist'
    data_flag = 'retinamnist'
    # data_flag = 'pneumoniamnist'
    # data_flag = 'dermamnist'
    download = True

    # NUM_EPOCHS = 3
    BATCH_SIZE = 64
    learning_rate = 0.001
    save_path = './_Save_ckpt{}Net.pth'.format(data_flag)
    epochs = 100
    best_acc = 0.0
    patience = 50

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])



    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[.5], std=[.5])
                                     ]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[.5], std=[.5])
                                   ])}

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform["train"], download=download, size=224, mmap_mode='r')
    val_dataset = DataClass(split='val', transform=data_transform["val"], download=download, size=224, mmap_mode='r')
  

    # encapsulate data into dataloader form
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    net = medmamba(in_chans=n_channels, num_classes=n_classes)

    if os.path.exists(save_path):
      net.load_state_dict(torch.load(save_path))
      print("Loaded existing checkpoints")

    net.to(device)
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    # macs, params = get_model_complexity_info(net, (3,224, 224), as_strings=True, backend='pytorch',
    #                                        print_per_layer_stat=True, verbose=True)
    # print(('Computational complexity: ', macs))
    # print(('Number of parameters: ', params))
    
    # exit(0)
    if task == "multi-label, binary-class":
      loss_function = nn.BCEWithLogitsLoss()
    else:
      loss_function = nn.CrossEntropyLoss()
      
    optimizer = optim.SGD(net.parameters(), learning_rate, momentum = 0.937, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose='deprecated')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= T_max_value)

    
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    counter = 0
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
                loss = loss_function(outputs, labels.to(device))
            else:
                labels = labels.squeeze().long()
                loss = loss_function(outputs, labels.to(device))

            loss.backward()
            optimizer.step()
        

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        y_true = torch.tensor([]).to(device)
        y_score = torch.tensor([]).to(device)
        split = 'test'
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                
                if task == 'multi-label, binary-class':
                    val_labels = val_labels.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    val_labels = val_labels.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    val_labels = val_labels.float().resize_(len(val_labels), 1)

                y_true = torch.cat((y_true, val_labels.to(device)), 0)
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.detach().cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            
            evaluator = Evaluator(data_flag, split)
            metrics = evaluator.evaluate(y_score)
            
            acc = metrics[1]
            auc = metrics[0]
            print(acc, auc)
            # exit(0)


        val_accurate = acc
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            counter = 0
            best_epoch_value = epoch + 1
            print('Validation accuracy increased from %.3f --> %.3f'% (best_acc,val_accurate))
            print('model saved at epoch:',best_epoch_value)
            
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        else:
          counter +=1
          if counter >= patience:
            print("Early stopping....The best model is saved at %.2f"%(best_epoch_value))
            break

    print('Finished Training....The best model is saved at %.2f'%(best_epoch_value))


if __name__ == '__main__':
    main()

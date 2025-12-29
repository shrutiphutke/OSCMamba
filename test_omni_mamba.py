import os
import sys
# import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
# import torch.optim as optim
from tqdm import tqdm
import math
from OmniMamba_v7 import VSSM as medmamba 
# import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # data_flag = 'pathmnist'
    # data_flag = 'octmnist'
    data_flag = 'retinamnist'
    # data_flag = 'pneumoniamnist'
    # data_flag = 'dermamnist'
    download = True

    # NUM_EPOCHS = 3
    BATCH_SIZE = 64
    net_path = './_Save_ckpt{}Net.pth'.format(data_flag)
    split = 'test'

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[.5], std=[.5])
                                   ])



    # load the data
    test_dataset = DataClass(split='test', transform=data_transform, download=download, size=224, mmap_mode='r')
    # pil_dataset = DataClass(split='test', download=download)
    # encapsulate data into dataloader form
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    net = medmamba(in_chans=n_channels,num_classes=n_classes)

    if os.path.exists(net_path):
      net.load_state_dict(torch.load(net_path))
      print("Loaded existing checkpoints")
    else:
        print("Provide the path of trained checkpoits")

    net.to(device)

   
    net.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)
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
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

    print('Finished Testing')


if __name__ == '__main__':
    main()

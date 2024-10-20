# """"
 
# ## update the comments here

# """



# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import config
# from tqdm import tqdm
# import numpy as np
# import pdb
# import os
# from natsort import natsorted
# import cv2
# import pickle
# from glob import glob
# from torch.utils.data import DataLoader
# # from pytorch_metric_learning import miners, losses

# import torch
# # import lpips
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from dataloader import EEGDataset
# from network import EEGFeatNet
# # from model import ModifiedResNet
# # from CLIPModel import CLIPModel
# # from visualizations import Umap, K_means, TsnePlot, save_image
# from losses import ContrastiveDynamicLoss
# # from dataaugmentation import apply_augmentation

# from datasets import load_dataset
# from torch.utils.data import random_split


# np.random.seed(45)
# torch.manual_seed(45)

# def train(epoch, model, optimizer, loss_fn, train_data, train_dataloader, experiment_num):
        
#     running_loss      = []
#     # eeg_featvec       = np.array([])
#     # eeg_featvec_proj  = np.array([])
#     # eeg_gamma         = np.array([])
#     # labels_array      = np.array([])

#     tq = tqdm(train_dataloader)
#     # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tq):
#     for batch_idx, (eegs,labels) in enumerate(tq, start=1):
#         # eeg_x1, eeg_x2 = eeg_x1.to(config.device), eeg_x2.to(config.device)

#         for idx,elem in enumerate(eegs):
#             eegs[idx] = elem.to(config.device)

#         for idx,elem in enumerate(labels):
#             labels[idx] = elem.to(config.device)

            

#         eeg_a , eeg_p, eeg_n = eegs



#         optimizer.zero_grad()

#         proj_a = model(eeg_a)
#         proj_p = model(eeg_p)
#         proj_n = model(eeg_n)
#         # print("proj_a.shape ",proj_a.shape)

#         projections_stack = torch.stack((proj_a,proj_p,proj_n), dim=1) 
#         labels_stack = torch.stack(labels,dim=1)

#         loss = loss_fn(projections_stack, labels_stack)
        
#         # loss  = loss_fn(x1_proj, x2_proj)
#         # backpropagate and update parameters
#         loss.backward()
#         optimizer.step()

#         running_loss = running_loss + [loss.detach().cpu().numpy()]

#         tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

#     # if (epoch%config.vis_freq) == 0:
#     #     # for batch_idx, (eeg, eeg_x1, eeg_x2, gamma, images, labels) in enumerate(tqdm(train_dataloader)):
#     #     for batch_idx, (eeg, images, labels, _) in enumerate(tqdm(train_dataloader)):
#     #         eeg, labels = eeg.to(config.device), labels.to(config.device)
#     #         with torch.no_grad():
#     #             x_proj = model(eeg)
#     #         # eeg_featvec      = np.concatenate((eeg_featvec, x.cpu().detach().numpy()), axis=0) if eeg_featvec.size else x.cpu().detach().numpy()
#     #         eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
#     #         # eeg_gamma        = np.concatenate((eeg_gamma, gamma.cpu().detach().numpy()), axis=0) if eeg_gamma.size else gamma.cpu().detach().numpy()
#     #         labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

#     #     ### compute k-means score and Umap score on the text and image embeddings
#     #     num_clusters   = config.num_classes
#     #     # k_means        = K_means(n_clusters=num_clusters)
#     #     # clustering_acc_feat = k_means.transform(eeg_featvec, labels_array)
#     #     # print("[Epoch: {}, Train KMeans score Feat: {}]".format(epoch, clustering_acc_feat))

#     #     k_means        = K_means(n_clusters=num_clusters)
#     #     clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
#     #     print("[Epoch: {}, Train KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

#     #     # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
#     #     # tsne_plot.plot(eeg_featvec, labels_array, clustering_acc_feat, 'train', experiment_num, epoch, proj_type='feat')
        

#     #     # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
#     #     # tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 'train', experiment_num, epoch, proj_type='proj')

#     return running_loss
 

# def validation(epoch, model, optimizer, loss_fn, miner, val_dataloader, experiment_num):
#     running_loss = []
#     eeg_featvec_proj = np.array([])
#     labels_array = np.array([])

#     tq = tqdm(val_dataloader)
#     for batch_idx, (eegs, labels) in enumerate(tq, start=1):
        
#         with torch.no_grad():
#             # Get the projections

#             for idx,elem in enumerate(eegs):
#                 eegs[idx] = elem.to(config.device)

#             for idx,elem in enumerate(labels):
#                 labels[idx] = elem.to(config.device)

#             eeg_a , eeg_p, eeg_n = eegs


#             proj_a = model(eeg_a)
#             proj_p = model(eeg_p)
#             proj_n = model(eeg_n)

#             # Stack projections
#             projections_stack = torch.stack((proj_a, proj_p, proj_n), dim=1) 
#             labels_stack = torch.stack(labels, dim=1)

#             # Calculate the loss
#             loss = loss_fn(projections_stack, labels_stack)

#             # Collect loss values
#             running_loss.append(loss.detach().cpu().numpy())

#         tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

#         # # Store projected features and labels for clustering
#         # eeg_featvec_proj = np.concatenate((eeg_featvec_proj, proj_a.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else proj_a.cpu().detach().numpy()
#         # labels_array = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

#     # Compute K-means score on the projected EEG features
#     # num_clusters = config.num_classes
#     # k_means = K_means(n_clusters=num_clusters)
#     # clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
#     # print("[Epoch: {}, Val KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

#     return running_loss
# # , clustering_acc_proj

    
# if __name__ == '__main__':

#     # base_path       = config.base_path
#     # train_path      = config.train_path
#     # validation_path = config.validation_path
#     device          = config.device
#     batch_size = config.batch_size
#     EPOCHS         = config.epoch




#     train_hf_ds = load_dataset("Alljoined/200sample")["train"]


#     train_data       = EEGDataset(train_hf_ds)

#     train_data,val_data = random_split(dataset=train_data,lengths=[0.8,0.20])
#     train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)


#     val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)



#     model     = EEGFeatNet( in_channels=config.input_size,\
#                            n_features=config.feat_dim,
#                              projection_dim=config.projection_dim,\
#                            num_layers=config.num_layers).to(config.device)
#     model     = torch.nn.DataParallel(model).to(config.device)
#     optimizer = torch.optim.Adam(\
#                                     list(model.parameters()),\
#                                     lr=config.lr,\
#                                     betas=(0.9, 0.999)
#                                 )

    
#     dir_info  = natsorted(glob('EXPERIMENT_*'))
#     if len(dir_info)==0:
#         experiment_num = 1
#     else:
#         experiment_num = int(dir_info[-1].split('_')[-1]) #+ 1

#     if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
#         os.makedirs('EXPERIMENT_{}'.format(experiment_num))
#         os.makedirs('EXPERIMENT_{}/val/tsne'.format(experiment_num))
#         os.makedirs('EXPERIMENT_{}/train/tsne/'.format(experiment_num))
#         os.makedirs('EXPERIMENT_{}/test/tsne/'.format(experiment_num))
#         os.makedirs('EXPERIMENT_{}/test/umap/'.format(experiment_num))
#         os.makedirs('EXPERIMENT_{}/finetune_ckpt/'.format(experiment_num))
#         os.makedirs('EXPERIMENT_{}/finetune_bestckpt/'.format(experiment_num))
#         os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))

#     ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/eegfeat_*.pth'.format(experiment_num)))

#     START_EPOCH = 0

#     if len(ckpt_lst)>=1:
#         ckpt_path  = ckpt_lst[-1]
#         checkpoint = torch.load(ckpt_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         START_EPOCH = checkpoint['epoch']
#         print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
#         START_EPOCH += 1
#     else:
#         os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
#         os.makedirs('EXPERIMENT_{}/bestckpt/'.format(experiment_num))

#     # miner   = miners.MultiSimilarityMiner()
#     loss_fn = ContrastiveDynamicLoss()
#     # loss_fn = ContrastiveLoss(batch_size=config.batch_size, temperature=config.temperature)
#     # loss_fn = PerceptualLoss()
#     # loss_fn   = F.l1_loss
#     # loss_fn = lpips.LPIPS(net='vgg').to(config.device)
#     # loss_fn  = nn.MSELoss()
#     # loss_fn  = nn.CrossEntropyLoss()
#     # base_eeg, base_images, base_labels, base_spectrograms = next(iter(val_dataloader))
#     # base_eeg, base_images = base_eeg.to(config.device), base_images.to(config.device)
#     # base_labels, base_spectrograms = base_labels.to(config.device), base_spectrograms.to(config.device)
#     best_val_acc   = 0.0
#     best_val_loss = -1 * np.inf
#     best_val_epoch = 0

#     for epoch in range(START_EPOCH, EPOCHS):

#         running_train_loss = train(epoch, model, optimizer, loss_fn,  train_data, train_dataloader, experiment_num)
#         if (epoch%config.vis_freq) == 0:
#         	running_val_loss   = validation(epoch, model, optimizer, loss_fn,  train_data, val_dataloader, experiment_num)

#         if best_val_acc < val_acc:
#         	best_val_acc   = val_acc
#         	best_val_epoch = epoch
#         	torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 # 'scheduler_state_dict': scheduler.state_dict(),
#               }, 'EXPERIMENT_{}/bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, 'all', val_acc))


#         torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 # 'scheduler_state_dict': scheduler.state_dict(),
#               }, 'EXPERIMENT_{}/checkpoints/eegfeat_{}.pth'.format(experiment_num, 'all'))

#         # running_val_loss   = validation(epoch, model, optimizer, loss_fn, train_data, val_dataloader)
#         # print(np.mean(running_train_loss), eeg_featvec.shape, labels_array.shape)

#         # if (epoch%1) == 0:
#         #     ### compute k-means score and Umap score on the text and image embeddings
#         #     num_clusters = 40
#         #     k_means        = K_means(n_clusters=num_clusters)
#         #     clustering_acc = k_means.transform(eeg_featvec, labels_array)
#         #     print("KMeans score:", clustering_acc)

#         #     with torch.no_grad():
#         #         pred = model(base_spectrograms)[0]
#         #         gt   = base_spectrograms[0]

#         #     save_image(pred, gt, experiment_num, epoch, 'val')
#         # break
#         # validate(model, 0.1, train_data)

#         # print('completed')
## update the comments here
"""
This script trains and validates an EEG feature extraction model using a contrastive loss function.
It saves checkpoints based on the evaluation loss instead of accuracy.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import config
from tqdm import tqdm
import numpy as np
import pdb
import os
from natsort import natsorted
import cv2
import pickle
from glob import glob
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import EEGDataset
from network import EEGFeatNet
from losses import ContrastiveDynamicLoss
from datasets import load_dataset
from torch.utils.data import random_split

np.random.seed(45)
torch.manual_seed(45)

def train(epoch, model, optimizer, loss_fn, train_data, train_dataloader, experiment_num):
    running_loss = []
    tq = tqdm(train_dataloader)
    
    for batch_idx, (eegs, labels) in enumerate(tq, start=1):
        for idx, elem in enumerate(eegs):
            eegs[idx] = elem.to(config.device)

        for idx, elem in enumerate(labels):
            labels[idx] = elem.to(config.device)

        eeg_a, eeg_p, eeg_n = eegs
        optimizer.zero_grad()

        proj_a = model(eeg_a)
        proj_p = model(eeg_p)
        proj_n = model(eeg_n)

        projections_stack = torch.stack((proj_a, proj_p, proj_n), dim=1)
        labels_stack = torch.stack(labels, dim=1)

        loss = loss_fn(projections_stack, labels_stack)
        
        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()

        running_loss.append(loss.detach().cpu().numpy())
        tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    return running_loss

def validation(epoch, model, optimizer, loss_fn, val_dataloader, experiment_num):
    running_loss = []
    tq = tqdm(val_dataloader)
    
    for batch_idx, (eegs, labels) in enumerate(tq, start=1):
        with torch.no_grad():
            for idx, elem in enumerate(eegs):
                eegs[idx] = elem.to(config.device)

            for idx, elem in enumerate(labels):
                labels[idx] = elem.to(config.device)

            eeg_a, eeg_p, eeg_n = eegs
            proj_a = model(eeg_a)
            proj_p = model(eeg_p)
            proj_n = model(eeg_n)

            projections_stack = torch.stack((proj_a, proj_p, proj_n), dim=1)
            labels_stack = torch.stack(labels, dim=1)

            loss = loss_fn(projections_stack, labels_stack)
            running_loss.append(loss.detach().cpu().numpy())

        tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    return running_loss

if __name__ == '__main__':
    device = config.device
    batch_size = config.batch_size
    EPOCHS = config.epoch

    train_hf_ds = load_dataset("Alljoined/200sample")["train"]
    train_data = EEGDataset(train_hf_ds)
    train_data, val_data = random_split(dataset=train_data, lengths=[0.8, 0.20])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    model = EEGFeatNet(in_channels=config.input_size,
                       n_features=config.feat_dim,
                       projection_dim=config.projection_dim,
                       num_layers=config.num_layers).to(config.device)
    model = torch.nn.DataParallel(model).to(config.device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.lr, betas=(0.9, 0.999))

    dir_info = natsorted(glob('EXPERIMENT_*'))
    experiment_num = 1 if len(dir_info) == 0 else int(dir_info[-1].split('_')[-1])  # + 1

    if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
        os.makedirs('EXPERIMENT_{}'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/val/tsne'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/train/tsne/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/test/tsne/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/test/umap/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/finetune_ckpt/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/finetune_bestckpt/'.format(experiment_num))
        os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))

    ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/eegfeat_*.pth'.format(experiment_num)))
    START_EPOCH = 0

    if len(ckpt_lst) >= 1:
        ckpt_path = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
        START_EPOCH += 1
    else:
        os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/bestckpt/'.format(experiment_num))

    loss_fn = ContrastiveDynamicLoss()
    
    best_val_loss = np.inf  # Initialize with infinity for loss comparison
    best_val_epoch = 0

    for epoch in range(START_EPOCH, EPOCHS):
        running_train_loss = train(epoch, model, optimizer, loss_fn, train_data, train_dataloader, experiment_num)
        
        if (epoch % config.vis_freq) == 0:
            running_val_loss = validation(epoch, model, optimizer, loss_fn, val_dataloader, experiment_num)
            avg_val_loss = np.mean(running_val_loss)  # Calculate average validation loss

            # Save the model if the validation loss is the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'EXPERIMENT_{}/bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, 'all', best_val_loss))

        # Save the current model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'EXPERIMENT_{}/checkpoints/eegfeat_{}.pth'.format(experiment_num, 'all'))


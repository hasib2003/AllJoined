import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
import config
import cv2
from dataaugmentation import apply_augmentation, extract_freq_band
import json

######### -----------------------------

from datasets import load_dataset

######### -----------------------------




class EEGDataset(Dataset):
    # def __init__(self, eegs, images, labels, subjects=None, n_fft=64, win_length=64, hop_length=16):
    def __init__(self,dataset):

        """"
        dataset: huggingface dataset object for Alljoined Dataset
        """

        # -----------------------------------------------

        print("# loading raw all joined dataset")

        self.categories = ['accessory',
                            'animal',
                            'appliance',
                            'electronic',
                            'food',
                            'furniture',
                            'indoor',
                            'kitchen',
                            'outdoor',
                            'person',
                            'sports',
                            'vehicle']
        
        with open("./annotations_trainval2017/annotations/instances_val2017.json","r") as f:
            instance_file = json.load(f)

        self.coco_annotations = instance_file
        
        self.cat2idx = {}
        for idx,val in enumerate(self.categories):
            self.cat2idx[val] = idx

        # print("self.cat2idx ",self.cat2idx)

        self.alljoined_dataset = dataset
        
        self.eegs   = []
        self.labels = [] # based on the coco ids

        for sample in tqdm(self.alljoined_dataset):
            # print("sample ",sample)

            classVector = [0 for i in self.categories]

            eeg = torch.tensor(np.array(sample["EEG"]).astype(np.float32)[:int(config.input_size),:].T)
            coco_id = int(sample["coco_id"])


            cats = self.get_image_supercategories(image_id=coco_id)
            # print("categories ",cats)

            for className in cats:
                idx = self.cat2idx[className]
                classVector[idx] = 1   

            # print("classVector ",classVector)         

            self.eegs.append(eeg)
            self.labels.append(torch.tensor(classVector))

        

        # creating a unifrom random set of indices 

        self.indicies = np.arange(self.__len__())
        self.indicies = np.random.permutation(self.indicies)
        
    
        # -----------------------------------------------

    ## thanks to chat gpt
    def get_image_supercategories(self, image_id):
        """
        Given the COCO dataset annotations and an image_id, returns the list of supercategories for that image.

        :param coco_annotations: Dictionary loaded from the COCO annotations JSON file.
        :param image_id: The ID of the image to find the supercategories for.
        :return: A set of supercategories for the image.
        """
        # Extract relevant data from the annotations
        annotations = self.coco_annotations['annotations']
        categories = self.coco_annotations['categories']

        # Create a map of category_id to supercategory for easy lookup
        supercategory_map = {cat['id']: cat['supercategory'] for cat in categories}

        # Find all category_ids for the given image_id
        category_ids = set()
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                category_ids.add(annotation['category_id'])  # Change here to get category_id

        # Convert category_ids to supercategories
        supercategories = {supercategory_map[cat_id] for cat_id in category_ids if cat_id in supercategory_map}

        return supercategories


    def getEEGInterface(self,index):

        return self.eegs[index],self.eegs[self.indicies[index]],self.eegs[(self.indicies[index]+1) % self.__len__()]


    def getLabelInterface(self,index):

        return self.labels[index],self.labels[self.indicies[index]],self.labels[(self.indicies[index]+1) % self.__len__()]




    def __getitem__(self, index):
     
        # eeg    = np.float32(self.eegs[index].cpu())
        # norm   = torch.max(eeg) / 2.0
        # eeg    = (eeg - norm)/ norm

        # print("torch.max(eeg) ",torch.max(eeg))


        ### one way could be to ensure that the anchor,positive and negative never matches
        
        # anchor,positive,negative

        return self.getEEGInterface(index),self.getLabelInterface(index)

        

    def __len__(self):
        return len(self.eegs)



if __name__ == '__main__':

    train_hf_ds = load_dataset("Alljoined/200sample")["train"]

    train_data  = EEGDataset(train_hf_ds)

    print("train_data[0] ",train_data[0])
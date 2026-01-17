import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count

TRAINING_SET_SIZE_PERCENT = 0.85
NO_CUDA = False

class ScheduleEvaluator:
    def __init__(self, model, schedule_dataset, batch_size):
        self.model = model
        self.schedule_dataset = schedule_dataset
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() and not NO_CUDA else 'cpu')
        self.model.to(self.device)

        self.schedule_train_dataset = None
        self.schedule_test_dataset = None

        print(f"Using device: {self.device}")
        self._split_datasets()

    def _split_datasets(self):
        '''
        Reproducible split of dataset into train and test sets for all schedule examples.
        '''
    
        if self.schedule_train_dataset is None or self.schedule_test_dataset is None:
            gen = torch.Generator().manual_seed(41)
            
            dataset_size = len(self.schedule_dataset)
            training_size = int(TRAINING_SET_SIZE_PERCENT * dataset_size)
            test_size = dataset_size - training_size
            
            train_set, test_set = random_split(
                self.schedule_dataset, 
                [training_size, test_size], 
                generator=gen
            )


            print(f"First element of the schedule training set: {train_set[0]}")
            print(f"First element of the schedule test set: {test_set[0]}")

            #save splitted datasets for future evaluation
            self.schedule_train_dataset = train_set
            self.schedule_test_dataset = test_set
        
    def weighted_loss(self, predictions, ground_truth, u_batch):
        """
        computes weighted BCE loss for activation, assignment, and sequence heads.
        total Loss = Beta * act_loss + alpha * (assign_loss + seq_loss)
        """
        pred_act = predictions['activation']
        pred_assign = predictions['assignment']
        pred_seq = predictions['sequence']
        
        #ground truth (should be in [N, 1] shape)
        true_act = ground_truth['operator'].y.view(-1, 1)
        true_assign = ground_truth['operator', 'assign', 'order'].y.view(-1, 1)
        true_seq = ground_truth['order', 'to', 'order'].y.view(-1, 1)
        
        #BCE losses
        loss_act = F.binary_cross_entropy(pred_act, true_act)
        loss_assign = F.binary_cross_entropy(pred_assign, true_assign)
        loss_seq = F.binary_cross_entropy(pred_seq, true_seq)
        
        #extract alpha/beta (Mmean over batch)
        alpha = u_batch[:, 0].mean()
        beta = u_batch[:, 1].mean()
        
        #weighted sum
        #Note that alpha/beta need to be scaled down if they are large (e.g. 100) to prevent explosion
        #or rely on the optimizer (Adam) to handle scaling.
        total_loss = (beta * loss_act) + (alpha * (loss_assign + loss_seq))
        
        return total_loss, loss_act.item(), loss_assign.item(), loss_seq.item()
    
    def evaluate(self, use_train_set=False):
        '''
        evaluates the model on the training/test dataset and returns average loss. 
        use_train_set: if True, trains and evaluates on training set, else on test set.
        '''

        schedule_dataset = self.schedule_train_dataset if use_train_set else self.schedule_test_dataset

        self.model.eval()
        data_loader = DataLoader(schedule_dataset, batch_size=self.batch_size, shuffle=False)
        total_epoch_loss = 0.0
        act_loss = 0.0
        assign_loss = 0.0
        seq_loss = 0.0

        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating on {'training' if use_train_set else 'test'} set"):
            batch = batch.to(self.device)
            
            #construct batch_dict
            batch_dict_arg = {
                'operator': batch['operator'].batch,
                'order': batch['order'].batch
            }
            
            #forward pass
            preds = self.model(
                batch.x_dict, 
                batch.edge_index_dict, 
                batch.edge_attr_dict,
                batch.u,
                batch_dict=batch_dict_arg
            )
            
            loss, l_act, l_assign, l_seq = self.weighted_loss(preds, batch, batch.u)
            total_epoch_loss += loss.item()
            act_loss += l_act
            assign_loss += l_assign
            seq_loss += l_seq

        average_total_loss = total_epoch_loss / len(schedule_dataset)
        average_act_loss = act_loss / len(schedule_dataset)
        average_assign_loss = assign_loss / len(schedule_dataset)
        average_seq_loss = seq_loss / len(schedule_dataset)

        return average_total_loss, average_act_loss, average_assign_loss, average_seq_loss
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
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
        total Loss = beta * act_loss + alpha * (assign_loss + seq_loss)

        We should not artificially force the final weighted loss to be between 0 and 1 
        (e.g., by passing it through a sigmoid or clamping it).
        Doing so would distort or kill the gradients, making the training impossible.

        However, we must control the magnitude of the weights ($\alpha, \beta$) 
        to prevent the loss from becoming too large (exploding gradients) 
        or too small (vanishing gradients). They got normalized in the data building stage.
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
        
        #extract alpha/beta (mean over batch)
        alpha = u_batch[:, 0].mean()
        beta = u_batch[:, 1].mean()
        
        #weighted sum
        #Note that alpha/beta need to be scaled down if they are large (e.g. 100) to prevent explosion
        #or rely on the optimizer (Adam) to handle scaling.
        total_loss = (beta * loss_act) + (alpha * (loss_assign + loss_seq))
        
        #avoid division by zero
        #sum_weights = alpha + beta + 1e-6
        #normalized_loss = total_loss / sum_weights

        return total_loss, loss_act.item(), loss_assign.item(), loss_seq.item()

    def calculate_metrics(self, preds, batch):
        """
        Calculates accuracy and confusion matrix for Activation, Assignment, and Sequence.
        
        Args:
            preds (dict): Output from model(batch) containing 'activation', 'assignment', 'sequence'
                        These are ALREADY probabilities (0-1) due to sigmoid in model.
            batch (HeteroData): The batch containing ground truth labels.
            
        Returns:
            dict: Contains accuracy and confusion matrix for each head.
        """
        metrics = {}
        
        #threshold for binary classification like logistic regression after sigmoid
        threshold = 0.5

        #activation head (operator nodes)
        if 'activation' in preds:
            #preds shape: [num_operators, 1]
            y_prob = preds['activation'].detach().cpu().numpy()
            y_pred = (y_prob > threshold).astype(int).flatten()
            
            #operator ground truth: batch['operator'].y
            y_true = batch['operator'].y.detach().cpu().numpy().flatten()
                
            metrics['act_acc'] = accuracy_score(y_true, y_pred)
            metrics['act_cm'] = confusion_matrix(y_true, y_pred, labels=[0, 1])

        #assignment head (operator -> order edges)
        if 'assignment' in preds:
            #preds shape: [num_assign_edges, 1]
            y_prob = preds['assignment'].detach().cpu().numpy()
            y_pred = (y_prob > threshold).astype(int).flatten()
            
            #assignment ground truth: ['operator', 'assign', 'order'].y
            y_true = batch['operator', 'assign', 'order'].y.detach().cpu().numpy().flatten()

            metrics['assign_acc'] = accuracy_score(y_true, y_pred)
            metrics['assign_cm'] = confusion_matrix(y_true, y_pred, labels=[0, 1])

        #sequence head (order -> order edges)
        if 'sequence' in preds:
            #preds shape: [num_seq_edges, 1]
            y_prob = preds['sequence'].detach().cpu().numpy()
            y_pred = (y_prob > threshold).astype(int).flatten()

            #sequence ground truth: ['order', 'to', 'order'].y
            y_true = batch['order', 'to', 'order'].y.detach().cpu().numpy().flatten()

            metrics['seq_acc'] = accuracy_score(y_true, y_pred)
            metrics['seq_cm'] = confusion_matrix(y_true, y_pred, labels=[0, 1])

        return metrics

    def calculate_f1_metrics(self, cm):
        """
        Calculates Precision, Recall, and F1 from a 2x2 confusion matrix.
        cm format: [[TN, FP], [FN, TP]]
        """
        tn, fp, fn, tp = cm.ravel()
        
        #precision: TP / (TP + FP)
        #if TP+FP is 0 (no positive predictions), precision is undefined (we use 0.0 instead)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        #recall: TP / (TP + FN)
        #if TP+FN is 0 (no actual positives), recall is undefined (we use 0.0 instead)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        #f1 score: 2 * (P * R) / (P + R)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        #support: total actual positives (TP + FN)
        support = tp + fn

        return precision, recall, f1, support


    def evaluate(self, use_train_set=False):
        '''
        evaluates the model on the training/test dataset and returns average loss. 
        use_train_set: if true, trains and evaluates on training set, else on test set.
        '''

        schedule_dataset = self.schedule_train_dataset if use_train_set else self.schedule_test_dataset

        self.model.eval()
        data_loader = DataLoader(schedule_dataset, batch_size=self.batch_size, shuffle=False)
        total_epoch_loss = 0.0
        total_epoch_accuracy = 0.0

        #single heads performance
        act_loss = 0.0
        assign_loss = 0.0
        seq_loss = 0.0
        act_accuracy = 0.0
        assign_accuracy = 0.0
        seq_accuracy = 0.0     
        act_cm = np.zeros((2, 2), dtype=int)
        assign_cm = np.zeros((2, 2), dtype=int)
        seq_cm = np.zeros((2, 2), dtype=int)

        with torch.no_grad():
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
                measurements = self.calculate_metrics(preds, batch)
                total_epoch_loss += loss.item()
                total_epoch_accuracy += sum([measurements['act_acc'], measurements['assign_acc'], measurements['seq_acc']]) / 3.0
    
                #accumulate single head losses
                act_loss += l_act
                assign_loss += l_assign
                seq_loss += l_seq

                #accumulate single head accuracies
                act_accuracy += measurements['act_acc']
                assign_accuracy += measurements['assign_acc']
                seq_accuracy += measurements['seq_acc']

                #accumulate confusion matrices
                act_cm += measurements['act_cm']
                assign_cm += measurements['assign_cm']
                seq_cm += measurements['seq_cm']

            #compute average losses
            average_total_loss = total_epoch_loss / len(data_loader)
            average_act_loss = act_loss / len(data_loader)
            average_assign_loss = assign_loss / len(data_loader)
            average_seq_loss = seq_loss / len(data_loader)

            #compute average accuracies
            average_total_accuracy = total_epoch_accuracy / len(data_loader)
            average_act_accuracy = act_accuracy / len(data_loader)
            average_assign_accuracy = assign_accuracy / len(data_loader)
            average_seq_accuracy = seq_accuracy / len(data_loader)

            #compute confusion matrix
            total_cm = act_cm + assign_cm + seq_cm
            row_sums = total_cm.sum(axis=1, keepdims=True)
            total_normalized_cm = total_cm / row_sums 

            return {
                'total_loss': average_total_loss,
                'act_loss': average_act_loss,
                'assign_loss': average_assign_loss,
                'seq_loss': average_seq_loss,
                'total_accuracy': average_total_accuracy,
                'act_accuracy': average_act_accuracy,
                'assign_accuracy': average_assign_accuracy,
                'seq_accuracy': average_seq_accuracy,
                'act_cm': act_cm,
                'assign_cm': assign_cm,
                'seq_cm': seq_cm,
                'total_cm': total_cm
            }
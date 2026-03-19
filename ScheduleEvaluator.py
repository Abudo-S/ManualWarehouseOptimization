import torch
import random
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
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
NUM_AUGMENTATIONS = 4 #number of new spatially augmented copies to make per original graph

#default threshold for binary classification accurcy like logistic regression after sigmoid
#need to be tuned if the classes are imbalanced (can be relevated from classification report / roc curve)
CLASSIFICATION_THRESHOLD = 0.05

class ScheduleEvaluator:
    def __init__(self, 
                 model, 
                 schedule_dataset, 
                 batch_size,
                 act_threshold=CLASSIFICATION_THRESHOLD,
                 assign_threshold=CLASSIFICATION_THRESHOLD,
                 seq_threshold=CLASSIFICATION_THRESHOLD,
                 split_train_validation=True,
                 use_spatial_augmentation=True,
                 n_spatial_augmentations_per_graph=NUM_AUGMENTATIONS):
        """
        Initializes the ScheduleEvaluator with the model, dataset, and evaluation parameters.
        Args:
            model (nn.Module): The GNN model to be evaluated.
            schedule_dataset (Dataset): The dataset containing schedule examples.
            batch_size (int): Batch size for evaluation.
            act_threshold (float): Threshold for activation head.
            assign_threshold (float): Threshold for assignment head.
            seq_threshold (float): Threshold for sequence head.

        If split_train_validation is False, all data examples are saved into schedule_val_dataset. (Note that large scale batches aren't labelled)
        """

        self.model = model
        self.schedule_dataset = schedule_dataset
        self.batch_size = batch_size
        self.act_threshold = act_threshold
        self.assign_threshold = assign_threshold
        self.seq_threshold = seq_threshold
        self.use_spatial_augmentation = use_spatial_augmentation
        self.n_spatial_augmentations_per_graph = n_spatial_augmentations_per_graph

        self.device = torch.device('cuda' if torch.cuda.is_available() and not NO_CUDA else 'cpu')
        self.model.to(self.device)

        self.schedule_train_dataset = None
        self.schedule_val_dataset = None

        print(f"Using device: {self.device}")

        if split_train_validation:
            self._split_datasets()
        else: #test set
            self.schedule_val_dataset = schedule_dataset

        if use_spatial_augmentation and split_train_validation:
            self.schedule_train_dataset = self.augment_dataset(self.schedule_train_dataset)
        elif use_spatial_augmentation: #isomorphic test-time augmentation
            #self.augment_dataset(self.schedule_val_dataset, n_spatial_augmentations_per_graph) #tta
            self.schedule_val_dataset = self.augment_isomorphic_dataset(self.schedule_val_dataset)

    def _split_datasets(self):
        '''
        Reproducible split of dataset into train and validation sets for all schedule examples.
        '''
    
        if self.schedule_train_dataset is None or self.schedule_val_dataset is None:
            gen = torch.Generator().manual_seed(41)
            
            dataset_size = len(self.schedule_dataset)
            training_size = int(TRAINING_SET_SIZE_PERCENT * dataset_size)
            val_size = dataset_size - training_size
            
            train_set, val_set = random_split(
                self.schedule_dataset, 
                [training_size, val_size], 
                generator=gen
            )

            print(f"First element of the schedule training set: {train_set[0]}")
            print(f"First element of the schedule validation set: {val_set[0]}")

            #save splitted datasets for future evaluation
            self.schedule_train_dataset = train_set
            self.schedule_val_dataset = val_set

    def augment_dataset(self, schedule_dataset):
        expanded_train_dataset = []

        print(f"Augmenting dataset from {len(schedule_dataset)} examples...")

        for i in range(len(schedule_dataset)):
            original_data = schedule_dataset[i]
            
            #add the unaltered original graph to our new dataset
            expanded_train_dataset.append(original_data)
            
            #generate and add the augmented copies
            for _ in range(self.n_spatial_augmentations_per_graph):
                augmented_data = self.augment_single_graph(original_data)
                expanded_train_dataset.append(augmented_data)

        print(f"Augmentation complete! New training dataset size: {len(expanded_train_dataset)}")

        return expanded_train_dataset

    def augment_dataset(self, schedule_dataset):
        expanded_train_dataset = []

        print(f"Augmenting dataset from {len(schedule_dataset)} examples...")

        for i in range(len(schedule_dataset)):
            original_data = schedule_dataset[i]
            
            #add the unaltered original graph to our new dataset
            expanded_train_dataset.append(original_data)
            
            #generate and add the augmented copies
            for _ in range(self.n_spatial_augmentations_per_graph):
                augmented_data = self.augment_single_graph(original_data)
                expanded_train_dataset.append(augmented_data)

        print(f"Augmentation complete! New training dataset size: {len(expanded_train_dataset)}")

        return expanded_train_dataset
    
    def augment_isomorphic_dataset(self, dataset):
        """
        Takes a dataset (list of HeteroData test batches) and returns a new 
        expanded dataset containing all isomorphic variations.
        """
        expanded_dataset = []
        
        for i, batch in enumerate(dataset):
            variants = self.generate_isomorphic_test_suite(batch, batch_id=i+1)
            expanded_dataset.extend(variants)
            
        return expanded_dataset

    def augment_single_graph(self, data):
        """
        Takes a single HeteroData graph, clones it, and applies spatial augmentations (translation, mirroring and rotation)..
        Perserving physical distance between nodes.
        Expects order features where indices 4-9 are: FROM_X, FROM_Y, FROM_Z, TO_X, TO_Y, TO_Z
        """
        aug_data = data.clone()
        
        #flip x-axis randomly (50% chance)
        if random.random() > 0.5:
            aug_data['order'].x[:, 4] = -aug_data['order'].x[:, 4] #FROM_X
            aug_data['order'].x[:, 7] = -aug_data['order'].x[:, 7] #TO_X

        #flip y-axis randomly (50% chance)
        if random.random() > 0.5:
            aug_data['order'].x[:, 5] = -aug_data['order'].x[:, 5] #FROM_Y
            aug_data['order'].x[:, 8] = -aug_data['order'].x[:, 8] #TO_Y

        #random Translation (shift coordinates)
        shift_x = random.uniform(-20.0, 20.0)
        shift_y = random.uniform(-20.0, 20.0)
        
        aug_data['order'].x[:, 4] += shift_x #FROM_X
        aug_data['order'].x[:, 7] += shift_x #TO_X
        
        aug_data['order'].x[:, 5] += shift_y #FROM_Y
        aug_data['order'].x[:, 8] += shift_y #TO_Y
        
        #swap x and y axes randomly
        if random.random() > 0.5:
            temp_from_x = aug_data['order'].x[:, 4].clone()
            aug_data['order'].x[:, 4] = aug_data['order'].x[:, 5]
            aug_data['order'].x[:, 5] = temp_from_x
            
            temp_to_x = aug_data['order'].x[:, 7].clone()
            aug_data['order'].x[:, 7] = aug_data['order'].x[:, 8]
            aug_data['order'].x[:, 8] = temp_to_x

        return aug_data

    def generate_isomorphic_test_suite(self, original_batch, batch_id):
        """
        Takes a single original HeteroData test batch and generates 9 specific 
        distance-preserving geometric variations. 
        Returns a list of 10 batches (1 original + 9 variants).
        """
        test_suite = []
        
        #original
        orig = original_batch.clone()
        orig.variant_name = f"Batch {batch_id}: original"
        test_suite.append(orig)
        
        #flip x-axis
        flip_x = original_batch.clone()
        flip_x['order'].x[:, 4] = -flip_x['order'].x[:, 4]
        flip_x['order'].x[:, 7] = -flip_x['order'].x[:, 7]
        flip_x.variant_name = f"Batch {batch_id}: flipped x-axis"
        test_suite.append(flip_x)

        #flip y-axis
        flip_y = original_batch.clone()
        flip_y['order'].x[:, 5] = -flip_y['order'].x[:, 5]
        flip_y['order'].x[:, 8] = -flip_y['order'].x[:, 8]
        flip_y.variant_name = f"Batch {batch_id}: flipped y-axis"
        test_suite.append(flip_y)

        #swap x and y (equivalent to a diagonal flip / 90-degree rotation shift)
        swap_xy = original_batch.clone()
        temp_from_x = swap_xy['order'].x[:, 4].clone()
        swap_xy['order'].x[:, 4] = swap_xy['order'].x[:, 5]
        swap_xy['order'].x[:, 5] = temp_from_x
        temp_to_x = swap_xy['order'].x[:, 7].clone()
        swap_xy['order'].x[:, 7] = swap_xy['order'].x[:, 8]
        swap_xy['order'].x[:, 8] = temp_to_x
        swap_xy.variant_name = f"Batch {batch_id}: swapped x & y"
        test_suite.append(swap_xy)

        #define a set of specific translations (shifts)
        #adjust these numbers based on the scale of your warehouse map (e.g., meters)
        shifts = [
            ("Shifted +20X, +20Y", 20.0, 20.0),
            ("Shifted -20X, -20Y", -20.0, -20.0),
            ("Shifted +50X, -50Y", 50.0, -50.0),
            ("Shifted -50X, +50Y", -50.0, 50.0),
            ("Extreme shift +200X", 200.0, 0.0),
            ("Extreme shift +200Y", 0.0, 200.0)
        ]

        for name, shift_x, shift_y in shifts:
            shifted = original_batch.clone()
            shifted['order'].x[:, 4] += shift_x #FROM_X
            shifted['order'].x[:, 7] += shift_x #TO_X
            shifted['order'].x[:, 5] += shift_y #FROM_Y
            shifted['order'].x[:, 8] += shift_y #TO_Y
            shifted.variant_name = f"Batch {batch_id}: {name}"
            test_suite.append(shifted)

        return test_suite

    def weighted_loss(self, 
                      predictions, 
                      ground_truth, 
                      u_batch,
                      act_loss_weight=1.0,
                      assign_loss_weight=1.0,
                      seq_loss_weight=1.0,
                      capacity_penalty_weight=1.5,
                      heuristic_boost_factor=1.15):
        """
        computes weighted BCE loss for activation, assignment, and sequence heads.
        total Loss = beta * act_loss + alpha * (assign_loss + seq_loss) + capacity_penalty

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
        
        # #capacity penalty: penalizes the model if expected workload of an operator exceeds h_fixed
        
        p_assign_flat = pred_assign.squeeze() #[num_edges]
        src_idx = ground_truth['operator', 'assign', 'order'].edge_index[0] #source operator indices
        proc_times = ground_truth['operator', 'assign', 'order'].edge_attr[:, 0] #processing times
        
        #calculate expected workload per operator: sum(p_assign * processing_time)
        num_ops_total = ground_truth['operator'].x.size(0)
        expected_workloads = torch.zeros(num_ops_total, device=pred_assign.device)

        #not fully accurate since it doesn't consider the exact travel time
        expected_workloads.scatter_add_(0, src_idx, p_assign_flat * proc_times * heuristic_boost_factor)

        #get h_fixed per operator
        #since u is [batch_size, 3] and h_fixed is at index 2
        op_batch = ground_truth['operator'].batch if hasattr(ground_truth['operator'], 'batch') else torch.zeros(num_ops_total, dtype=torch.long, device=pred_assign.device)
        h_fixed_per_op = ground_truth.u[op_batch, 2]

        #penalize workload that exceeds h_fixed using relu (0 if workload < h_fixed)
        capacity_violations = torch.relu(expected_workloads - h_fixed_per_op)

        #only average the penalty over operators that exceed capacity
        active_violations = capacity_violations > 0
        if active_violations.sum() > 0:
            capacity_penalty = capacity_violations.sum() / active_violations.sum()
        else:
            capacity_penalty = torch.tensor(0.0, device=pred_assign.device)

        #weighted sum
        #Note that alpha/beta need to be scaled down if they are large (e.g. 100) to prevent explosion
        #or rely on the optimizer (Adam) to handle scaling.
        #total_loss = (beta * loss_act) + (alpha * (loss_assign + loss_seq))
        # total_loss = (beta * (act_loss_weight * loss_act)) + \
        #              (alpha * ((assign_loss_weight * loss_assign) + (seq_loss_weight * loss_seq))) + \
        #              (capacity_penalty_weight * capacity_penalty)

        #multiplying by alpha and beta causes gradient vanishing for heads affected by the lower weight (alpha or beta)
        total_loss = (act_loss_weight * loss_act) + \
             (assign_loss_weight * loss_assign) + \
             (seq_loss_weight * loss_seq) + \
             (capacity_penalty_weight * capacity_penalty)
        
        #avoid division by zero
        #sum_weights = alpha + beta + 1e-6
        #normalized_loss = total_loss / sum_weights

        return total_loss, loss_act.item(), loss_assign.item(), loss_seq.item()

    def diagnose_operator_embeddings(self, model, batch):
        with torch.no_grad():
            #raw features
            x_dict = {'order': model.order_lin(batch.x_dict['order']).relu(),
                    'operator': model.op_lin(batch.x_dict['operator']).relu()}
            
            #message passing
            for conv in model.convs:
                x_dict = conv(x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                x_dict = {k: v.relu() for k, v in x_dict.items()}
            
            op_emb = x_dict['operator']
            
            print("------------Operator embedding diagnostics------------")
            print(f"Shape: {op_emb.shape}")
            print(f"Mean: {op_emb.mean():.4f}")
            print(f"Std:  {op_emb.std():.4f}")
            print(f"Min:  {op_emb.min():.4f}")
            print(f"Max:  {op_emb.max():.4f}")
            
            #check pairwise differences
            if op_emb.size(0) > 1:
                pairwise_diffs = []
                for i in range(op_emb.size(0)):
                    for j in range(i+1, op_emb.size(0)):
                        diff = (op_emb[i] - op_emb[j]).abs().mean()
                        pairwise_diffs.append(diff.detach().cpu().numpy())
                print(f"Mean pairwise diff: {np.mean(pairwise_diffs):.6f}")
                print(f"Max pairwise diff: {np.max(pairwise_diffs):.6f}")
            
            #check if all embeddings are nearly identical
            if op_emb.std() < 1e-4 or np.mean(pairwise_diffs) < 1e-4:
                print("Critical: Operator embeddings are nearly identical!")
            
            return op_emb


    def calc_f1_metrics(self, preds, targets, head_name=None):
        """
        Calculates Precision, Recall, and F1 score for binary classification.
        Args:
            preds (torch.Tensor): Model predictions (probabilities).
            targets (torch.Tensor): Ground truth labels.
            head_name (str): Name of the head ('activation', 'assignment', 'sequence') to select threshold.
        Returns:
            dict: Contains precision, recall, and f1 score.
        """

        if head_name == 'activation':
            threshold = self.act_threshold
        elif head_name == 'assignment':
            threshold = self.assign_threshold
        elif head_name == 'sequence':
            threshold = self.seq_threshold
        else:
            threshold = CLASSIFICATION_THRESHOLD

        y_pred = (preds > threshold).int().cpu().numpy()
        y_true = targets.cpu().numpy()
        
        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    def calc_overall_metrics(self, preds_dict, batch):
        """
        Calculates metrics for each head individually and then averages them
        to get an overall model performance score.
        
        Args:
            preds_dict (dict): Dictionary of model predictions {'activation': tensor, ...}
            batch (HeteroData): Batch object containing ground truth labels.
            
        Returns:
            dict: {
                'overall_f1': float, 
                'overall_precision': float, 
                'overall_recall': float,
                'activation': dict,  # individual metrics
                'assignment': dict,
                'sequence': dict
            }
        """
        head_metrics = {}
        total_f1 = 0
        total_prec = 0
        total_rec = 0
        valid_heads = 0

        #activation head
        if 'activation' in preds_dict:
            targets = batch['operator'].y
            m = self.calc_f1_metrics(preds_dict['activation'], targets, head_name='activation')
            head_metrics['activation'] = m
            total_f1 += m['f1']
            total_prec += m['precision']
            total_rec += m['recall']
            valid_heads += 1

        #assignment head
        if 'assignment' in preds_dict:
            targets = batch['operator', 'assign', 'order'].y
            m = self.calc_f1_metrics(preds_dict['assignment'], targets, head_name='assignment')
            head_metrics['assignment'] = m
            total_f1 += m['f1']
            total_prec += m['precision']
            total_rec += m['recall']
            valid_heads += 1

        #sequence head
        if 'sequence' in preds_dict:
            targets = batch['order', 'to', 'order'].y
            m = self.calc_f1_metrics(preds_dict['sequence'], targets, head_name='sequence')
            head_metrics['sequence'] = m
            total_f1 += m['f1']
            total_prec += m['precision']
            total_rec += m['recall']
            valid_heads += 1

        #average
        if valid_heads > 0:
            overall_metrics = {
                'overall_f1': total_f1 / valid_heads,
                'overall_precision': total_prec / valid_heads,
                'overall_recall': total_rec / valid_heads
            }
        else:
            overall_metrics = {'overall_f1': 0.0, 'overall_precision': 0.0, 'overall_recall': 0.0}

        #merge individual head metrics into the result for detailed logging
        overall_metrics.update(head_metrics)
        
        return overall_metrics
    
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
        
        #activation head (operator nodes)
        if 'activation' in preds:
            #preds shape: [num_operators, 1]
            y_prob = preds['activation'].detach().cpu().numpy()
            y_pred = (y_prob > self.act_threshold).astype(int).flatten()
            
            #operator ground truth: batch['operator'].y
            y_true = batch['operator'].y.detach().cpu().numpy().flatten()
                
            metrics['act_acc'] = accuracy_score(y_true, y_pred)
            metrics['act_cm'] = confusion_matrix(y_true, y_pred, labels=[0, 1])

        #assignment head (operator -> order edges)
        if 'assignment' in preds:
            #preds shape: [num_assign_edges, 1]
            y_prob = preds['assignment'].detach().cpu().numpy()
            y_pred = (y_prob > self.assign_threshold).astype(int).flatten()
            
            #assignment ground truth: ['operator', 'assign', 'order'].y
            y_true = batch['operator', 'assign', 'order'].y.detach().cpu().numpy().flatten()

            metrics['assign_acc'] = accuracy_score(y_true, y_pred)
            metrics['assign_cm'] = confusion_matrix(y_true, y_pred, labels=[0, 1])

        #sequence head (order -> order edges)
        if 'sequence' in preds:
            #preds shape: [num_seq_edges, 1]
            y_prob = preds['sequence'].detach().cpu().numpy()
            y_pred = (y_prob > self.seq_threshold).astype(int).flatten()

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

        schedule_dataset = self.schedule_train_dataset if use_train_set else self.schedule_val_dataset

        self.model.eval()
        data_loader = DataLoader(schedule_dataset, batch_size=self.batch_size, shuffle=False)
        total_epoch_loss = 0.0
        total_epoch_accuracy = 0.0
        total_epoch_f1 = 0.0

        #single heads performance
        act_loss = 0.0
        assign_loss = 0.0
        seq_loss = 0.0
        act_accuracy = 0.0
        assign_accuracy = 0.0
        seq_accuracy = 0.0  
        act_f1 = 0.0
        assign_f1 = 0.0
        seq_f1 = 0.0   
        act_cm = np.zeros((2, 2), dtype=int)
        assign_cm = np.zeros((2, 2), dtype=int)
        seq_cm = np.zeros((2, 2), dtype=int)

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating on {'training' if use_train_set else 'validation'} set"):
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
                f1_measurements = self.calc_overall_metrics(preds, batch)

                total_epoch_loss += loss.item()
                total_epoch_accuracy += sum([measurements['act_acc'], measurements['assign_acc'], measurements['seq_acc']]) / 3.0
                total_epoch_f1 += f1_measurements['overall_f1']

                #accumulate single head losses
                act_loss += l_act
                assign_loss += l_assign
                seq_loss += l_seq

                #accumulate single head accuracies
                act_accuracy += measurements['act_acc']
                assign_accuracy += measurements['assign_acc']
                seq_accuracy += measurements['seq_acc']
                
                #accumulate single head f1 scores
                act_f1 += f1_measurements['activation']['f1']
                assign_f1 += f1_measurements['assignment']['f1']
                seq_f1 += f1_measurements['sequence']['f1']

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

            #compute average f1 scores
            average_total_f1 = total_epoch_f1 / len(data_loader)
            average_act_f1 = act_f1 / len(data_loader)
            average_assign_f1 = assign_f1 / len(data_loader)
            average_seq_f1 = seq_f1 / len(data_loader)

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
                'total_f1': average_total_f1,
                'act_f1': average_act_f1,
                'assign_f1': average_assign_f1,
                'seq_f1': average_seq_f1,
                'act_cm': act_cm,
                'assign_cm': assign_cm,
                'seq_cm': seq_cm,
                'total_cm': total_cm
            }
    
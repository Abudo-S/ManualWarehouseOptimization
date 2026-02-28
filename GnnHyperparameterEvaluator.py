import torch
import numpy as np
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from ScheduleEvaluator import ScheduleEvaluator
from MultiCriteriaGNNModel import MultiCriteriaGNNModel
from GnnScheduleDataset import GnnScheduleDataset

LARGE_SCALE_BATCH_NAME = "Batch1000M" #Batch1000M
TARGET_MINI_BATCH_SIZE = 10 #number of missions per mini-batch
LARGE_SCALE_MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}.csv"
PREPROCESSED_BATCH_DIR = f"./preprocessed/{LARGE_SCALE_BATCH_NAME}/Batch{TARGET_MINI_BATCH_SIZE}M_idx.xlsx" #idx to be replaced cluster idx
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts10W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = f"./schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
NUM_EPOCHS = 10
BATCH_SIZE = 16 #nice to be equal to 32 or 64 since we have small mini-batch instances
LEARNING_RATE = 0.001

#default threshold for binary classification accurcy like logistic regression after sigmoid
#need to be tuned if the classes are imbalanced (can be relevated from classification report / roc curve)
CLASSIFICATION_THRESHOLD = 0.05
N_FOLDS = 5 #we can do 5-fold CV for hyperparameter tuning since we have a large number of mini-batch instances, but for final evaluation we will keep a separate test set apart (no data leakage)

class GnnHyperparameterEvaluator(ScheduleEvaluator):
    def __init__(self, 
                 model, #initial model (not used, will re-initialize per fold)
                 schedule_dataset, 
                 batch_size, #default batch size (not used, will be read from config)
                 learning_rate=0.001, #default lr (not used, will be read from config)
                 n_epochs=50,
                 default_threshold=CLASSIFICATION_THRESHOLD,
                 tune_multiple_thresholds=False): #whether to tune separate thresholds per head or a single shared one
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(self.device)

        self.dataset = schedule_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.default_threshold = default_threshold
        self.tune_multiple_thresholds = tune_multiple_thresholds

        super().__init__(model, schedule_dataset, batch_size)


    def train_and_evaluate_single_optimizer(self, config, dataset, k_folds=N_FOLDS):
        """
        Standard approach: Single lr for the entire model.
        Returns:
            avg_score (float): Average F1 score across folds
            avg_thresh (float or dict): Average best threshold(s) across folds
        """

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        print(f"Starting {k_folds}-Fold CV (single optimizer)({'separate-thresholds' if self.tune_multiple_thresholds else 'single-threshold'})...")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            #data splitting
            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

            #model initialization
            metadata = (dataset[0].metadata()[0], dataset[0].metadata()[1])
            model = MultiCriteriaGNNModel(
                metadata=metadata, 
                hidden_dim=config['hidden_dim'], 
                heads=config['heads']
            ).to(self.device)
            
            #single optimizer for all parameters
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            
            #training loop
            for epoch in range(self.n_epochs):
                model.train()
                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    
                    #batch_dict for the model
                    batch_dict_arg = {'operator': batch['operator'].batch, 'order': batch['order'].batch}
                    
                    preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.u, batch_dict=batch_dict_arg)
                    
                    #calculate loss
                    loss, l_act, l_assign, l_seq = self.weighted_loss(preds, batch, batch.u) 
                    
                    loss.backward()
                    optimizer.step()

            #evaluation & threshold tuning
            if self.tune_multiple_thresholds:
                best_f1, best_thresh = self._evaluate_fold_separate_thresholds(model, val_loader)
            else:
                best_f1, best_thresh = self._evaluate_fold(model, val_loader)
            
            fold_results.append({'val_score': best_f1, 'best_threshold': best_thresh})
            
            thresh_str = str(best_thresh) if isinstance(best_thresh, float) else str({k: round(v,3) for k,v in best_thresh.items()})
            print(f"Fold {fold+1}/{k_folds} | F1: {best_f1:.4f} | Thresh: {thresh_str}")

        #average results
        avg_score = sum(r['val_score'] for r in fold_results) / k_folds
        
        if self.tune_multiple_thresholds:
            avg_thresh = {}
            keys = fold_results[0]['best_threshold'].keys()
            for k in keys:
                avg_thresh[k] = sum(r['best_threshold'][k] for r in fold_results) / k_folds
        else:
            avg_thresh = sum(r['best_threshold'] for r in fold_results) / k_folds
        
        return avg_score, avg_thresh

    def train_and_evaluate_multi_optimizer(self, config, dataset, k_folds=N_FOLDS):
        """
        Advanced approach: Separate lrs for trunk and each head using separate optimizers.
        Returns:
            avg_score (float): Average F1 score across folds
            avg_thresh (float or dict): Average best threshold(s) across folds
        """
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        print(f"Starting {k_folds}-Fold CV (multi-optimizer)({'separate-thresholds' if self.tune_multiple_thresholds else 'single-threshold'})...")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            #data splitting
            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

            #model initialization
            metadata = (dataset[0].metadata()[0], dataset[0].metadata()[1])
            model = MultiCriteriaGNNModel(
                metadata=metadata, 
                hidden_dim=config['hidden_dim'], 
                heads=config['heads']
            ).to(self.device)

            #disjoint parameter definition
            trunk_params = (
                list(model.order_lin.parameters()) +
                list(model.op_lin.parameters()) +
                list(model.convs.parameters())
            )
            activation_params = list(model.activation_head.parameters())
            assignment_params = list(model.assign_head.parameters())
            sequence_params = list(model.seq_head.parameters())

            #separate optimizers
            #note: We get the config by key or provide its default value
            optimizers = []
            opt_trunk = torch.optim.Adam(trunk_params, lr=config.get('lr_trunk', self.learning_rate))
            opt_activation = torch.optim.Adam(activation_params, lr=config.get('lr_activation', self.learning_rate))
            opt_assignment = torch.optim.Adam(assignment_params, lr=config.get('lr_assignment', self.learning_rate))
            opt_sequence = torch.optim.Adam(sequence_params, lr=config.get('lr_sequence', self.learning_rate))
            optimizers.extend([opt_trunk, opt_activation, opt_assignment, opt_sequence])

            #training loop
            for epoch in range(self.n_epochs):
                model.train()
                for batch in train_loader:
                    batch = batch.to(self.device)
                    
                    #zero all gradients
                    for opt in optimizers:
                        opt.zero_grad()
                    
                    batch_dict_arg = {'operator': batch['operator'].batch, 'order': batch['order'].batch}
                    preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.u, batch_dict=batch_dict_arg)
                    
                    #total loss
                    loss, l_act, l_assign, l_seq = self.weighted_loss(preds, batch, batch.u)
                    
                    #backward pass for all parameters (trunk + heads)
                    loss.backward()

                    #step all optimizers independently
                    for opt in optimizers:
                        opt.step()

            #evaluation & threshold tuning
            if self.tune_multiple_thresholds:
                best_f1, best_thresh = self._evaluate_fold_separate_thresholds(model, val_loader)
            else:
                best_f1, best_thresh = self._evaluate_fold(model, val_loader)
            
            fold_results.append({'val_score': best_f1, 'best_threshold': best_thresh})
            
            thresh_str = str(best_thresh) if isinstance(best_thresh, float) else str({k: round(v,3) for k,v in best_thresh.items()})
            print(f"Fold {fold+1}/{k_folds} | F1: {best_f1:.4f} | Thresh: {thresh_str}")

        avg_score = sum(r['val_score'] for r in fold_results) / k_folds
        
        if self.tune_multiple_thresholds:
            avg_thresh = {}
            keys = fold_results[0]['best_threshold'].keys()
            for k in keys:
                avg_thresh[k] = sum(r['best_threshold'][k] for r in fold_results) / k_folds
        else:
            avg_thresh = sum(r['best_threshold'] for r in fold_results) / k_folds
        
        return avg_score, avg_thresh

    def _evaluate_fold_separate_thresholds(self, model, val_loader):
        """
        Evaluates the model and finds the best threshold SEPARATELY for each head
        using the Precision-Recall Curve (no manual loop needed).
        
        Returns: 
            avg_f1 (float): Average F1 across all heads (for hyperparameter selection)
            best_thresholds (dict): ex. {'activation': 0.4, 'assignment': 0.1, 'sequence': 0.3}
        """
        model.eval()
        
        #data to be stored per head
        head_data = {
            'activation': {'probs': [], 'labels': []},
            'assignment': {'probs': [], 'labels': []},
            'sequence': {'probs': [], 'labels': []}
        }
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                batch_dict_arg = {'operator': batch['operator'].batch, 'order': batch['order'].batch}
                preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.u, batch_dict=batch_dict_arg)

                for head in head_data:
                    if head in preds:
                        probs = preds[head].cpu().numpy().flatten()
                        
                        if head == 'activation':
                            labels = batch['operator'].y.cpu().numpy().flatten()
                        elif head == 'assignment':
                            labels = batch['operator', 'assign', 'order'].y.cpu().numpy().flatten()
                        elif head == 'sequence':
                            labels = batch['order', 'to', 'order'].y.cpu().numpy().flatten()
                        
                        head_data[head]['probs'].append(probs)
                        head_data[head]['labels'].append(labels)

        final_thresholds = {}
        final_f1s = {}
        
        for head, data in head_data.items():
            if not data['probs']: continue 
            
            y_scores = np.concatenate(data['probs'])
            y_true = np.concatenate(data['labels'])
            
            #vectorized optimal threshold search
            #precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            #note: thresholds array is shorter than p/r arrays by 1
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
            
            #calculate F1 for all thresholds at once
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
                f1_scores = np.nan_to_num(f1_scores) #replace NaNs with 0
            
            #idx of max F1
            best_idx = np.argmax(f1_scores)
            
            #handle edge case where best_idx might be the last element (threshold = 1.0)
            if best_idx < len(thresholds):
                best_t = thresholds[best_idx]
                best_f1 = f1_scores[best_idx]
            else:
                best_t = self.default_threshold #default fallback
                best_f1 = 0.0
            
            final_thresholds[head] = float(best_t)
            final_f1s[head] = float(best_f1)

        #metric for hyperparameter selection: average F1 of all heads
        avg_f1 = sum(final_f1s.values()) / len(final_f1s) if final_f1s else 0.0
        
        return avg_f1, final_thresholds

    def _evaluate_fold(self, model, val_loader):
        """
        Helper to run evaluation on a fold and tune threshold without retraining.
        Best threshold is found through precision-recall curve by maximizing F1 over all predictions in the validation set.
          
        Returns: best_f1, best_threshold
        """
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                batch_dict_arg = {'operator': batch['operator'].batch, 'order': batch['order'].batch}
                preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.u, batch_dict=batch_dict_arg)

                #aggregate predictions and labels from all heads
                for head_name in ['activation', 'assignment', 'sequence']:
                    if head_name in preds:
                        probs = preds[head_name].cpu().numpy().flatten()
                        
                        if head_name == 'activation':
                            labels = batch['operator'].y.cpu().numpy().flatten()
                        elif head_name == 'assignment':
                            labels = batch['operator', 'assign', 'order'].y.cpu().numpy().flatten()
                        elif head_name == 'sequence':
                            labels = batch['order', 'to', 'order'].y.cpu().numpy().flatten()
                        
                        all_probs.append(probs)
                        all_labels.append(labels)

        #concatenate all batches
        y_scores = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)

        #vectorized F1 calculation for all thresholds
        precisions, recalls, thresh_curve = precision_recall_curve(y_true, y_scores)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            f1_scores = np.nan_to_num(f1_scores)
        
        #find best F1
        #precisions/recalls have 1 extra element (for threshold=1.0)
        best_idx = np.argmax(f1_scores)
        
        if best_idx < len(thresh_curve):
            best_thresh = thresh_curve[best_idx]
            best_f1 = f1_scores[best_idx]
        else:
            best_thresh = self.default_threshold #default fallback
            best_f1 = 0.0

        return best_f1, best_thresh

    #wrapper to select train_and_evaluate method
    def train_and_evaluate(self, config, dataset, k_folds=N_FOLDS):
        if 'lr_trunk' in config:
            return self.train_and_evaluate_multi_optimizer(config, dataset, k_folds)
        else:
            return self.train_and_evaluate_single_optimizer(config, dataset, k_folds)
    
    
    def run_kfold_grid_search(self, dataset, param_grid, k_folds=N_FOLDS, start_config_num=1, min_f1=0.0):
        """
        Returns:
            best_config, best_threshold, best_score
        """

        #generate all combinations of hyperparameters
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"Total configurations to test: {len(combinations)}")
        
        best_score = -1.0
        best_config = None
        best_threshold = self.default_threshold #default init
        
        results_log = []

        #iterate through grid
        for i, config in enumerate(combinations):
            if i < start_config_num: #used to continue a previous execution (fact: itertools.product generate the same order of combinations each time)
                continue

            print(f"\n---Running config {i+1}/{len(combinations)}---")
            print(config)
            
            try:
                #the evaluator automatically chooses single vs multi optimizer based on keys in config
                avg_f1, avg_thresh = self.train_and_evaluate(
                    config=config, 
                    dataset=dataset, 
                    k_folds=k_folds
                )
                
                if isinstance(avg_thresh, dict):
                    thresh_str = str({k: round(v, 4) for k, v in avg_thresh.items()})
                else:
                    thresh_str = f"{avg_thresh:.4f}"
                    
                print(f"Result: avg F1 = {avg_f1:.4f} (best threshold: {thresh_str})")
                
                #log results
                results_log.append({
                    'config': config,
                    'score': avg_f1,
                    'threshold': avg_thresh
                })
                
                #update best config
                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_config = config
                    best_threshold = avg_thresh

                    if best_score > min_f1:
                        min_f1 = best_score
                        print(">>>New best configuration found!<<<")
                    
            except Exception as e:
                print(f"Error running config {config}: {str(e)}")
                continue

        print("\n-------------------FINAL RESULTS--------------------")
        print(f"Best F1 score: {best_score:.4f}")
        
        if isinstance(best_threshold, dict):
             thresh_str_final = str({k: round(v, 4) for k, v in best_threshold.items()})
        else:
             thresh_str_final = f"{best_threshold:.4f}"
        
        print(f"Best threshold: {thresh_str_final}")
        
        print("Best configuration:")
        if best_config:
            for k, v in best_config.items():
                print(f"{k}: {v}")
        
        return best_config, best_threshold, best_score

    def tune_threshold(self, target_head=None):
        """
        Finds the optimal classification threshold using the Precision-Recall Curve.
        The tuning is done on the validation set (here, we use the validation set for simplicity, but anyway the final test set is kept apart).
        Note that the threshold might need to be tuned separately for each head (activation, assignment, sequence).
        Here, we aggregate all predictions and labels from all heads and tune a shared single threshold for simplicity.
        
        -target_head (head_name or None): If specified, only tune threshold for this head. If None, aggregate all heads together for tuning (overall threshold tuning).

        Returns: best_threshold, best_f1
        """

        print(f"Tuning {target_head if target_head else 'overall'} threshold using validation Set...")
        
        #collect all probs and labels 
        all_probs = []
        all_labels = []
        
        self.model.eval()
        loader = DataLoader(self.schedule_val_dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                batch_dict_arg = {'operator': batch['operator'].batch, 'order': batch['order'].batch}
                preds = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.u, batch_dict=batch_dict_arg)
                
                #aggregate from all heads (activation, assignment, sequence) if target_head is none.
                #note: we need to tune heads separately for better performance per head
                target_heads = [target_head] if target_head else ['activation', 'assignment', 'sequence']
                for head_name in target_heads:
                    if head_name in preds:
                        #probs (sigmoid output)
                        probs = preds[head_name].cpu().numpy().flatten()
                        
                        #ground truth
                        #map head name to edge type/node type
                        if head_name == 'activation':
                            labels = batch['operator'].y.cpu().numpy().flatten()
                        elif head_name == 'assignment':
                            labels = batch['operator', 'assign', 'order'].y.cpu().numpy().flatten()
                        elif head_name == 'sequence':
                            labels = batch['order', 'to', 'order'].y.cpu().numpy().flatten()
                            
                        all_probs.append(probs)
                        all_labels.append(labels)

        #concatenate everything into one giant array
        y_scores = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)
        
        #thresholds array is one element smaller than precision/recall arrays
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        #calculate F1 for every single threshold
        #handle division by zero (0 precision + 0 recall)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores) #replace NaNs with 0
        
        #find the index of the best F1
        #Note: f1_scores has same length as precisions/recalls, which is len(thresholds) + 1
        #We ignore the last element (which corresponds to threshold=1.0)
        best_idx = np.argmax(f1_scores)
        
        #safety check if best_idx is out of bounds for thresholds
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
        else:
            best_threshold = self.threshold  #fallback
            best_f1 = 0.0

        print(f"Optimal threshold: {best_threshold:.4f} (max F1: {best_f1:.4f})")
        
        return best_threshold, best_f1

if __name__ == "__main__":
    #init dataset
    dataset = GnnScheduleDataset(
        schedule_dir=SCHEDULE_DIR,
        mission_base_path=MISSION_BATCH_DIR,
        edge_base_path=MISSION_BATCH_TRAVEL_DIR,
        pallet_types_file_path=UDC_TYPES_DIR,
        fork_path=FORK_LIFTS_DIR
    )

    #hyperparameter space
    # param_grid = {
    #     'batch_size': [32, 64, 128], #we have small graphs, so we can try larger batch sizes
    #     'learning_rate': [0.01, 0.001, 0.0005], 
    #     'hidden_dim': [64, 128], #GNN hidden dimension
    #     'heads': [4, 8] #GATv2 heads
    # }

    # possibile thresholds (cheap evaluation, no retraining) 
    # note that the best is found automatically via precision-recall curve
    # thresholds = np.linspace(0.01, 0.5, 50).tolist() #[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5]

    #multi-optimizer grid (separate lrs)
    param_grid_multi = {
        'batch_size': [32, 64],
        'hidden_dim': [64, 128],
        'heads': [4], #GAT heads
        'lr_trunk': [1e-3, 5e-4], #backbone lr
        'lr_activation': [1e-2, 1e-3], #head 1 lr
        'lr_assignment': [1e-3], #head 2 lr
        'lr_sequence': [1e-3]  #head 3 lr
    }
    
    #single optimizer grid (global lr)
    # param_grid_single = {
    #     'batch_size': [32, 64],
    #     'hidden_dim': [64],
    #     'heads': [4],
    #     'learning_rate': [0.01, 0.001] #global LR
    # }

    #init model
    if len(dataset) > 0:
        sample_data = dataset[0]
        model = MultiCriteriaGNNModel(
            metadata=sample_data.metadata(),
            hidden_dim=64,
            num_layers=3,
            heads=4
        )

    gnnHyperparamEvaluator = GnnHyperparameterEvaluator(model=model, 
                                                        schedule_dataset=dataset,   
                                                        batch_size=BATCH_SIZE,
                                                        learning_rate=LEARNING_RATE,
                                                        n_epochs=NUM_EPOCHS,
                                                        tune_multiple_thresholds=True) #tune separate threshold per head
    
    best_conf, best_val = gnnHyperparamEvaluator.run_kfold_grid_search(dataset, param_grid_multi)
    
    print("Best hyperparameter configuration from grid search:")
    print(best_conf)
    
import torch
import numpy as np
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from ScheduleEvaluator import ScheduleEvaluator
from MultiCriteriaGNNModel import MultiCriteriaGNNModel

class GnnHyperparameterEvaluator(ScheduleEvaluator):
    def __init__(self, 
                 model, #initial model (not used, will re-initialize per fold)
                 schedule_dataset, 
                 batch_size,
                 learning_rate=0.001, #default lr (not used, will get from config)
                 n_epochs=50):
        
        super().__init__(model, schedule_dataset, batch_size)
        self.dataset = schedule_dataset
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_and_evaluate_single_optimizer(self, config, dataset, thresholds, k_folds=5):
        """
        Standard approach: Single lr for the entire model.
        """

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        print(f"Starting {k_folds}-Fold CV (single optimizer)...")

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
                    loss_dict = self.weighted_loss(preds, batch, batch.u) 
                    loss = loss_dict['total_loss']
                    
                    loss.backward()
                    optimizer.step()

            #evaluation & threshold Tuning
            best_f1, best_thresh = self._evaluate_fold(model, val_loader, thresholds)
            fold_results.append({'val_score': best_f1, 'best_threshold': best_thresh})
            print(f"Fold {fold+1}/{k_folds} | F1: {best_f1:.4f} | Thresh: {best_thresh:.4f}")

        #average results
        avg_score = sum(r['val_score'] for r in fold_results) / k_folds
        avg_thresh = sum(r['best_threshold'] for r in fold_results) / k_folds
        
        return avg_score, avg_thresh

    def train_and_evaluate_multi_optimizer(self, config, dataset, thresholds, k_folds=5):
        """
        Advanced approach: Separate lrs for trunk and each Head using separate optimizers.
        """
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        print(f"Starting {k_folds}-Fold CV (multi-optimizer)...")

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
                    loss_dict = self.weighted_loss(preds, batch, batch.u)
                    loss = loss_dict['total_loss']
                    
                    #backward pass for all parameters (trunk + heads)
                    loss.backward()

                    #step all optimizers independently
                    for opt in optimizers:
                        opt.step()

            #evaluation
            best_f1, best_thresh = self._evaluate_fold(model, val_loader, thresholds)
            fold_results.append({'val_score': best_f1, 'best_threshold': best_thresh})
            print(f"Fold {fold+1}/{k_folds} | F1: {best_f1:.4f} | Threshold: {best_thresh:.4f}")

        avg_score = sum(r['val_score'] for r in fold_results) / k_folds
        avg_thresh = sum(r['best_threshold'] for r in fold_results) / k_folds
        
        return avg_score, avg_thresh

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
            best_thresh = 0.5
            best_f1 = 0.0

        return best_f1, best_thresh

    #wrapper to select train_and_evaluate method
    def train_and_evaluate(self, config, dataset, k_folds=5):
        if 'lr_trunk' in config:
            return self.train_and_evaluate_multi_optimizer(config, dataset, k_folds)
        else:
            return self.train_and_evaluate_single_optimizer(config, dataset, k_folds)
    
    
    def run_grid_search(self, dataset, param_grid, k_folds=5):
        #initialize evaluator (Model is re-init inside, so we pass None or a dummy)
        #we pass a dummy batch_size initially, it gets overridden by config
        evaluator = GnnHyperparameterEvaluator(model=None, schedule_dataset=dataset, batch_size=32, n_epochs=20)

        #generate all combinations of hyperparameters
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"Total configurations to test: {len(combinations)}")
        
        best_score = -1.0
        best_config = None
        best_threshold = 0.5
        
        results_log = []

        #iterate through Grid
        for i, config in enumerate(combinations):
            print(f"\n--- Running Config {i+1}/{len(combinations)} ---")
            print(config)
            
            try:
                #the evaluator automatically chooses single vs multi optimizer based on keys in config
                avg_f1, avg_thresh = evaluator.train_and_evaluate(
                    config=config, 
                    dataset=dataset, 
                    k_folds=k_folds
                )
                
                print(f"Result: Avg F1 = {avg_f1:.4f} (Best Threshold: {avg_thresh:.4f})")
                
                #log results
                results_log.append({
                    'config': config,
                    'score': avg_f1,
                    'threshold': avg_thresh
                })
                
                #update best
                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_config = config
                    best_threshold = avg_thresh
                    print(">>> New Best Configuration Found! <<<")
                    
            except Exception as e:
                print(f"Error running config {config}: {str(e)}")
                continue

        print("\n-------------------FINAL RESULTS--------------------")
        print(f"Best F1 Score: {best_score:.4f}")
        print(f"Best Threshold: {best_threshold:.4f}")
        print("Best Configuration:")
        for k, v in best_config.items():
            print(f"{k}: {v}")
        
        return best_config, best_score

    def tune_threshold(self):
        """
        Finds the optimal classification threshold using the Precision-Recall Curve.
        The tuning is done on the validation set (here, we use the validation set for simplicity, but anyway the final test set is kept apart).
        Note that the threshold might need to be tuned separately for each head (activation, assignment, sequence).
        Here, we aggregate all predictions and labels from all heads and tune a shared single threshold for simplicity.
        
        Returns: best_threshold, best_f1
        """

        print("Tuning threshold using validation Set...")
        
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
                
                #aggregate from all heads (activation, assignment, sequence)
                #note: we might tune them separately for better performance per head
                for head_name in ['activation', 'assignment', 'sequence']:
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
        
        #use Scikit-Learn to get P/R curve
        #thresholds array is one element smaller than precision/recall arrays
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        #calculate F1 for every single threshold
        #handle division by zero (0 precision + 0 recall)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores) # Replace NaNs with 0
        
        #find the index of the best F1
        #Note: f1_scores has same length as precisions/recalls, which is len(thresholds) + 1
        #We ignore the last element (which corresponds to threshold=1.0)
        best_idx = np.argmax(f1_scores)
        
        #safety check if best_idx is out of bounds for thresholds
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
        else:
            best_threshold = self.threshold   #fallback
            best_f1 = 0.0

        print(f"Optimal threshold: {best_threshold:.4f} (max F1: {best_f1:.4f})")
        
        return best_threshold, best_f1

if __name__ == "__main__":
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
        'lr_trunk': [1e-3, 5e-4], #backbone LR
        'lr_activation': [1e-2, 1e-3], #head 1 LR
        'lr_assignment': [1e-3], #head 2 LR
        'lr_sequence': [1e-3]  #head 3 LR
    }
    
    #single optimizer grid (global lr)
    # param_grid_single = {
    #     'batch_size': [32, 64],
    #     'hidden_dim': [64],
    #     'heads': [4],
    #     'learning_rate': [0.01, 0.001] #global LR
    # }
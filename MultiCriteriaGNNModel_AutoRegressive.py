import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import HeteroConv, GATv2Conv, to_hetero
from torch_geometric.utils import softmax as tg_softmax
import os

SAVE_MODEL_PATH = "checkpoints/gnn_autoregressive_model_weights.pth"
SAVE_MODEL_IN_TRAINING_PATH = "checkpoints/gnn_autoregressive_checkpoint_epoch_idx.pth" #replace idx with epoch number when saving

class MultiCriteriaGNNModel_AutoRegressive(nn.Module):
    def __init__(self, hidden_dim=64, heads=4):
        super().__init__()
        
        #node features inputs:
        #operator: 7 physical + 6 AR dynamic features = 13 (PE is handled separately)
        self.op_lin = Linear(7 + 6, hidden_dim)
        
        #order: 10 original + 1 AR dynamic feature (is_assigned mask) = 11
        self.order_lin = Linear(10 + 1, hidden_dim)

        #message passing layers
        # self.convs = nn.ModuleList([
        #     GATv2Conv((-1, -1), hidden_dim, heads=num_heads, add_self_loops=False, edge_dim=2),
        #     GATv2Conv((-1, -1), hidden_dim, heads=num_heads, add_self_loops=False, edge_dim=2),
        #     GATv2Conv((-1, -1), hidden_dim, heads=num_heads, add_self_loops=False, edge_dim=1)
        # ])

        self.convs = nn.ModuleList()
        
        for _ in range(3): #3 message passing layers
            conv = HeteroConv({
                #note the addition of concat=False
                ('operator', 'assign', 'order'): GATv2Conv((-1, -1), hidden_dim, heads=heads, concat=False, add_self_loops=False, edge_dim=2),
                
                ('order', 'rev_assign', 'operator'): GATv2Conv((-1, -1), hidden_dim, heads=heads, concat=False, add_self_loops=False, edge_dim=2),
                
                ('order', 'to', 'order'): GATv2Conv((-1, -1), hidden_dim, heads=heads, concat=False, add_self_loops=False, edge_dim=1)
            }, aggr='mean')
            self.convs.append(conv)

        #output heads (predicts the next action)
        
        #action/assignment head 
        input_dim_action = 2 * hidden_dim + 3 + 2 + 8
        self.action_head = Sequential(
            Linear(input_dim_action, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

        #sequence head (predicts order -> order next step)
        #input: order_emb(src) + order_emb(dst) + global_u + edge_attr(travel_time)
        input_dim_seq = 2 * hidden_dim + 3 + 1
        self.seq_head = Sequential(
            Linear(input_dim_seq, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

        #termination Head 
        input_dim_term = hidden_dim + 3 + 8
        self.termination_head = Sequential(
            Linear(input_dim_term, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, u, batch_dict=None):
        raw_op_features = x_dict['operator']
        
        op_physical = raw_op_features[:, :7]
        if raw_op_features.size(1) >= 21:
            op_pe = raw_op_features[:, 7:15]
            op_dynamic = raw_op_features[:, 15:] 
        elif raw_op_features.size(1) == 13: 
            op_pe = torch.zeros((raw_op_features.size(0), 8), device=raw_op_features.device)
            op_dynamic = raw_op_features[:, 7:13]
        else:
            raise ValueError(f"Unexpected operator feature dimension: {raw_op_features.size(1)}")
            
        op_input = torch.cat([op_physical, op_dynamic], dim=1)
        x_dict_internal = {
            'operator': self.op_lin(op_input).relu(),
            'order': self.order_lin(x_dict['order']).relu()
        }

        orig_op_emb = x_dict_internal['operator'].clone()
        orig_ord_emb = x_dict_internal['order'].clone()

        for conv in self.convs:
            x_dict_internal = conv(x_dict_internal, edge_index_dict, edge_attr_dict)
            x_dict_internal = {key: torch.clamp(x, min=-1e4, max=1e4).relu() for key, x in x_dict_internal.items()}
            x_dict_internal = {key: torch.nan_to_num(x, nan=0.0) for key, x in x_dict_internal.items()}

        x_dict_internal['operator'] = 0.5 * orig_op_emb + 0.5 * x_dict_internal['operator']
        x_dict_internal['order'] = 0.5 * orig_ord_emb + 0.5 * x_dict_internal['order']

        if batch_dict is None:
            batch_dict = {key: torch.zeros(x.size(0), dtype=torch.long, device=x.device) for key, x in x_dict_internal.items()}
        op_batch = batch_dict['operator']
        order_batch = batch_dict['order']

        #termination Prediction
        term_input = torch.cat([x_dict_internal['operator'], u[op_batch], op_pe], dim=1)
        out_terminate = torch.sigmoid(self.termination_head(term_input))

        #assignment Prediction
        src_idx_assign, dst_idx_assign = edge_index_dict[('operator', 'assign', 'order')]
        edge_attr_assign = edge_attr_dict[('operator', 'assign', 'order')].view(-1, 2)
        action_input = torch.cat([
            x_dict_internal['operator'][src_idx_assign],
            x_dict_internal['order'][dst_idx_assign],
            u[op_batch[src_idx_assign]],
            edge_attr_assign,
            op_pe[src_idx_assign]
        ], dim=1)
        out_assign = torch.sigmoid(self.action_head(action_input))

        #sequence prediction (order -> order)
        src_idx_seq, dst_idx_seq = edge_index_dict[('order', 'to', 'order')]
        edge_attr_seq = edge_attr_dict[('order', 'to', 'order')].view(-1, 1)
        seq_input = torch.cat([
            x_dict_internal['order'][src_idx_seq],
            x_dict_internal['order'][dst_idx_seq],
            u[order_batch[src_idx_seq]],
            edge_attr_seq
        ], dim=1)
        out_seq = torch.sigmoid(self.seq_head(seq_input))

        return {
            'action': out_assign,      #[num_assign_edges, 1]
            'sequence': out_seq,       #[num_seq_edges, 1]
            'terminate': out_terminate #[num_ops, 1]
        }
    
    def save_model(self, save_path=SAVE_MODEL_PATH):
        print(f"Saving model weights to {save_path}...")

        #create checkpoints directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        #save model weights
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path=SAVE_MODEL_PATH, device='cuda'):
        print(f"Loading model weights from {save_path}...")
        
        #load model weights
        self.load_state_dict(torch.load(save_path, map_location=torch.device(device)))
        self.eval()

    def save_model_in_training(self,
                               optimizers:dict, 
                               current_epoch, 
                               current_loss, 
                               save_weights_path=SAVE_MODEL_PATH,
                               save_path=SAVE_MODEL_IN_TRAINING_PATH):
        """
        Saves model weights and training state (optimizer states, epoch, loss) to a checkpoint for resuming training later.
            optimizers: A dictionary of optimizers used in training (e.g., {'trunk_optimizer': optimizer}, ...).
            current_epoch: The current epoch number (used to identify the checkpoint file).
            current_loss: The current loss value (for logging and checkpointing).
            save_weights_path: Path to save the model weights file (used for saving the model architecture and weights).
            save_path: Path template for the training checkpoint (should include "idx" to be replaced with epoch number).
        """

        #save weights
        self.save_model(save_weights_path)

        #handle both single optimizer (legacy) and dictionary of optimizers (new)
        if isinstance(optimizers, dict):
            optimizer_state_dict = {k: v.state_dict() for k, v in optimizers.items()}
        else:
            optimizer_state_dict = optimizers.state_dict()

        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state_dict, #saves dict or single state
            'loss': current_loss,
            #save hyperparameters so they don't get forgetten them
            'hyperparameters': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'heads': self.heads
            }
        }

        #save training checkpoint
        torch.save(checkpoint, save_path.replace("idx", str(current_epoch)))

    def load_model_in_training(self,
                               current_epoch,
                               optimizers:dict,
                               save_weights_path=SAVE_MODEL_PATH,
                               save_path=SAVE_MODEL_IN_TRAINING_PATH):
        """
        Loads model weights and training state (optimizer states, epoch, loss) from a checkpoint for resuming training.
            current_epoch: The epoch number to load (used to identify the correct checkpoint file).
            optimizers: A dictionary of optimizers used in training (e.g., {'trunk_optimizer': optimizer}, ...).
            save_weights_path: Path to the model weights file (used for loading the model architecture and weights).
            save_path: Path template for the training checkpoint (should include "idx" to be replaced with epoch number).
        Returns: A tuple containing the optimizers with loaded states, the starting epoch for resuming training, and the loss value at the checkpoint.
        """

        #load weights
        self.load_model(save_weights_path)

        #load training checkpoint
        checkpoint = torch.load(save_path.replace("idx", str(current_epoch)))

        #retrieve states
        self.load_state_dict(checkpoint['model_state_dict'])
        saved_optimizer_states = checkpoint['optimizer_state_dict']

        #load optimizer states
        if isinstance(optimizers, dict):
            #verify the checkpoint contains a dictionary of states
            if not isinstance(saved_optimizer_states, dict):
                raise TypeError("Checkpoint contains a single optimizer state, but a dictionary of optimizers was provided.")
                
            for key, opt in optimizers.items():
                if key in saved_optimizer_states:
                    opt.load_state_dict(saved_optimizer_states[key])
                    print(f"Loaded state for optimizer: {key}")
                else:
                    print(f"Warning: Key '{key}' not found in checkpoint optimizer states.")
        else: #single optimizer
            optimizers.load_state_dict(saved_optimizer_states)


        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        self.train()

        return optimizers, start_epoch, loss




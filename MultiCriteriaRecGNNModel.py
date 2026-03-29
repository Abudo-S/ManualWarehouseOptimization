import math
import torch
import torch.nn.functional as F
from torch.nn import GRUCell
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import HeteroConv, GATv2Conv, to_hetero
from torch_geometric.utils import softmax as tg_softmax
import os

SAVE_MODEL_PATH = "checkpoints/rec_gnn_model_weights.pth"
SAVE_MODEL_IN_TRAINING_PATH = "checkpoints/rec_gnn_checkpoint_epoch_idx.pth" #replace idx with epoch number when saving

class MultiCriteriaRecGNNModel(torch.nn.Module):
    def __init__(self, 
                 metadata, 
                 hidden_dim=64, 
                 num_layers=3, 
                 heads=4, 
                 dropout=0.2,
                 head_coupling_dropout=0.0,
                 heuristic_boost_factor=1.15):
        '''
        metadata: Tuple of (node_types, edge_types) from the heterogeneous graph
        hidden_dim: Dimension of hidden embeddings
        num_layers: Number of GNN layers
        heads: Number of attention heads in GATv2
        dropout: Dropout rate for GATv2 layers to regularize the attention mechanism.
        head_coupling_dropout: Dropout rate for the coupling between heads. To avoid edge over-smoothing over sequence.
            If we assign head_coupling_dropout=1 (full decoupling), the model will not learn a clean, distinct "Active" vs "Inactive" feature.
            But the sequence head will return to outputting high-quality spatial sequences.
        heuristic_boost_factor: Used to travel+processing time estimation to give an initial start to activation head with minimum_ops. (ex. +15%)
            (e.g., max time in minutes) to [0,1] range for better training stability.
        defines a multi-criteria GNN model with three heads:
        1.Activation Head: Classifies operator nodes as active/inactive
        2.Assignment Head: Classifies edges from operators to orders
        3.Sequence Head: Classifies edges between orders

        Note that model time features (H_fixed, processing time, travel time) should be pre-scaled to 
        [0,1] range before being fed into the model for better training stability. 
        This can be done by dividing the raw time values by a constant (e.g., 480.0 minutes for an 8-hour shift) 
        to bring them to a similar scale as the node features.

        The Architecture: Static Encoder + Recurrent Decoder:
        1- The Encoder (Static): The GNN runs exactly once at the beginning. It looks at the warehouse and creates fixed embeddings for all orders.
        2- The Decoder (Recurrent): a GRUCell specifically for the operators.
           - the hidden_state of the GRU represents the operator's current status (location, remaining capacity, etc.).
           - when an operator finishes an order, we feed that order's embedding into the GRU. The GRU updates the operator's hidden_state.
           - we use new hidden_state and Global Fleet Context to predict the next assignment.
        '''
        super().__init__()
        assert heuristic_boost_factor >= 1.0, "heuristic_boost_factor should be >= 1.0"
        print(f"metadata: {metadata}")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.head_coupling_dropout = torch.nn.Dropout(head_coupling_dropout)
        self.heuristic_boost_factor = heuristic_boost_factor

        #STR- static encoder, should be executed once
        
        #orders: 10 physical features
        self.order_lin = Linear(10, hidden_dim) 
        
        #operators: 7 physical features + 8 PE = 15 dimensions
        self.op_lin = Linear(15, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ('order', 'to', 'order'): GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, add_self_loops=False, dropout=dropout), 
                ('operator', 'assign', 'order'): GATv2Conv((-1, -1), hidden_dim // heads, heads=heads, edge_dim=2, add_self_loops=False, dropout=dropout),
                ('order', 'rev_assign', 'operator'): GATv2Conv((-1, -1), hidden_dim // heads, heads=heads, edge_dim=2, add_self_loops=False, dropout=dropout)
            }
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
            
        #STR- recurrent decoder, should be executed n steps
        
        #GRU updates the operator's hidden state locally. 
        #last_order_emb (hidden_dim) + dynamic features (remaining_h, current_X, current_Y)
        self.op_rnn = GRUCell(input_size=hidden_dim + 3, hidden_size=hidden_dim)
        
        #activation head
        #RNN state (64) + FLEET CONTEXT (64) + global_u (3) + demand (1) + monotonic_id (1) + op_pe (8) = 141
        input_dim_activation = (2 * hidden_dim) + 3 + 1 + 1 + 8
        self.activation_head = Sequential(
            Linear(input_dim_activation, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

        #assignment head
        #RNN state (64) + FLEET CONTEXT (64) + order emb (64) + global_u (3) + time (2) + act_prob (1) + monotonic_id (1) + op_pe (8) + affinity_score (1) = 208
        input_dim_assignment = (3 * hidden_dim) + 3 + 2 + 1 + 1 + 8 + 1
        self.assign_head = Sequential(
            Linear(input_dim_assignment , hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

        #sequence head
        #ord_i (64) + ord_j (64) + global (3) + time (1) + shared_Op (1) + active_shared (1) = 134
        input_dim_sequence = (2 * hidden_dim) + 3 + 1 + 1 + 1
        self.seq_head = Sequential(
            Linear(input_dim_sequence, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

    def encode(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Step 1: Create static embeddings for the warehouse layout.
        Call this once before starting the routing loop.
        """
        #extract purely physical features (Indices 0-9 for orders, 0-14 for ops (physical + PE))
        order_physical = x_dict['order'][:, :10]
        op_physical_and_pe = x_dict['operator'][:, :15]

        #initial projection
        x_dict_static = {
            'order': self.order_lin(order_physical).relu(),
            'operator': self.op_lin(op_physical_and_pe).relu()
        }
        
        orig_op_emb = x_dict_static['operator'].clone()
        orig_ord_emb = x_dict_static['order'].clone()

        #message passing
        for conv in self.convs:
            x_dict_mp = conv(x_dict_static, edge_index_dict, edge_attr_dict)
            x_dict_static = {key: torch.clamp(x, min=-1e4, max=1e4).relu() for key, x in x_dict_mp.items()}
            x_dict_static = {key: torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4) for key, x in x_dict_static.items()}
        
        #residual connection
        x_dict_static['operator'] = 0.5 * orig_op_emb + 0.5 * x_dict_static['operator']
        x_dict_static['order'] = 0.5 * orig_ord_emb + 0.5 * x_dict_static['order']

        return x_dict_static

    def decode_step(self, static_embs, x_dict_raw, edge_index_dict, edge_attr_dict, u, batch_dict, op_hidden=None, last_order_emb=None):
        """
        Step 2: The Recurrent step.
        Updates the GRU state based on current capacity/location and predicts next assignments.
        """
        #extract features from raw inputs
        #operator PE (idx 7 to 15)
        op_pe = x_dict_raw['operator'][:, 7:15]

        #order dynamic: [is_assigned] (idx 10)
        is_assigned_flag = x_dict_raw['order'][:, 10].unsqueeze(1)
        
        #operator dynamic: [remaining_h_fixed, current_X, current_Y] (idx 15, 16, 17)
        op_dynamic = x_dict_raw['operator'][:, 15:18]
        remaining_h = op_dynamic[:, 0].unsqueeze(1)
        
        #initialize GRU states if this is the first step
        if op_hidden is None:
            op_hidden = static_embs['operator'] #start with static GNN embedding
        if last_order_emb is None:
            last_order_emb = torch.zeros_like(static_embs['operator']) #no previous order

        #update operator state via GRU
        rnn_input = torch.cat([last_order_emb, op_dynamic], dim=1)
        new_op_hidden = self.op_rnn(rnn_input, op_hidden)

        op_batch = batch_dict['operator']

        #global fleet context calculation
        #we average the updated hidden states of all operators in the same graph/batch
        num_graphs = u.size(0)
        fleet_sum = torch.zeros((num_graphs, self.hidden_dim), dtype=new_op_hidden.dtype, device=new_op_hidden.device)
        fleet_sum.index_add_(0, op_batch, new_op_hidden)
        
        fleet_count = torch.zeros((num_graphs, 1), dtype=new_op_hidden.dtype, device=new_op_hidden.device)
        fleet_count.index_add_(0, op_batch, torch.ones_like(op_batch, dtype=new_op_hidden.dtype).unsqueeze(1))
        
        fleet_mean = fleet_sum / fleet_count.clamp(min=1.0)
        global_fleet_context = fleet_mean[op_batch]  #broadcast back to ops

        #calculate monotonic ids
        num_ops_total = new_op_hidden.size(0)
        monotonic_id = torch.zeros((num_ops_total, 1), dtype=torch.float, device=new_op_hidden.device)
        for i in range(u.size(0)):
            mask = (op_batch == i)
            num_ops_in_graph = mask.sum()
            if num_ops_in_graph > 0:
                local_ids = torch.arange(num_ops_in_graph, dtype=torch.float, device=new_op_hidden.device) / num_ops_in_graph.float()
                monotonic_id[mask, 0] = local_ids
        
        u_ops = u[op_batch]

        #basic demand heuristic calculation
        assign_edges = edge_index_dict[('operator', 'assign', 'order')]
        proc_times = edge_attr_dict[('operator', 'assign', 'order')][:, 0]
        sum_proc_per_order = torch.zeros(static_embs['order'].size(0), dtype=proc_times.dtype, device=new_op_hidden.device)
        sum_proc_per_order.index_add_(0, assign_edges[1], proc_times)

        count_per_order = torch.zeros(static_embs['order'].size(0), dtype=proc_times.dtype, device=new_op_hidden.device)
        count_per_order.index_add_(0, assign_edges[1], torch.ones_like(proc_times))
        
        avg_proc_per_order = sum_proc_per_order / count_per_order.clamp(min=1.0)
        total_workload_per_batch = torch.zeros(u.size(0), dtype=proc_times.dtype, device=new_op_hidden.device)
        total_workload_per_batch.index_add_(0, batch_dict['order'], avg_proc_per_order)
        
        ops_per_batch = torch.zeros(u.size(0), dtype=torch.float, device=new_op_hidden.device)
        ops_per_batch.index_add_(0, op_batch, torch.ones_like(op_batch, dtype=torch.float))

        min_ops_needed = (total_workload_per_batch / (u[:, 2] + 1e-6)) * self.heuristic_boost_factor
        
        #calculate an activation threshold for each operator based on its monotonic id.
        op_ordinal_index = monotonic_id.squeeze() * ops_per_batch[op_batch]
        min_ops_broadcast = min_ops_needed[op_batch]
        
        #soft demand feature:
        #if ordinal < min_ops, it outputs a positive value representing "how much" it belongs in the active set.
        #if ordinal >= min_ops, it strongly outputs -1.0 to keep it inactive.
        op_demand_feature = torch.where(
            op_ordinal_index < min_ops_broadcast,
            1.0 - (op_ordinal_index / min_ops_broadcast.clamp(min=1.0)), #decays from 1.0 to 0.0 smoothly
            -1.0 #hard cutoff for unneeded operators
        )

        #activation head (using new GRU state + fleet_context + op_pe)
        op_feat_final = torch.cat([
            new_op_hidden, 
            global_fleet_context, 
            u_ops, 
            op_demand_feature.unsqueeze(1), 
            monotonic_id, 
            op_pe
        ], dim=1)
        out_activation = torch.sigmoid(self.activation_head(op_feat_final))

        #assignment head
        src_idx, dst_idx = edge_index_dict[('operator', 'assign', 'order')]
        op_emb_assign = new_op_hidden[src_idx]
        ord_emb_assign = static_embs['order'][dst_idx]
        edge_attr = edge_attr_dict[('operator', 'assign', 'order')]

        u_edges = u[op_batch[src_idx]]
        op_activation_prob = out_activation[src_idx].detach()
        src_monotonic_id = monotonic_id[src_idx]
        
        #affinity Score calculation
        raw_ord_x, raw_ord_y = x_dict_raw['order'][:, 4], x_dict_raw['order'][:, 5]
        ord_x_dest, ord_y_dest = raw_ord_x[dst_idx], raw_ord_y[dst_idx]
        ord_x_norm = ord_x_dest / (ord_x_dest.max() + 1e-6)
        ord_y_norm = ord_y_dest / (ord_y_dest.max() + 1e-6)
        order_angle = torch.atan2(ord_y_norm, ord_x_norm) / (math.pi / 2)
        affinity_score = torch.abs(src_monotonic_id.squeeze() - order_angle).unsqueeze(1)

        assign_input = torch.cat([
            op_emb_assign, 
            global_fleet_context[src_idx], #fleet_contextbroadcast to edges
            ord_emb_assign, 
            u_edges, 
            edge_attr, 
            op_activation_prob, 
            src_monotonic_id, 
            op_pe[src_idx], 
            affinity_score
        ], dim=1)

        out_assign = torch.sigmoid(self.assign_head(assign_input))

        #assignment masking: natively enforce constraints
        valid_assign_mask = ((1.0 - is_assigned_flag[dst_idx]) * (remaining_h[src_idx] > 0.0).float())
        out_assign = out_assign * valid_assign_mask
        
        #sequence head
        src_seq, dst_seq = edge_index_dict.get(('order', 'to', 'order'), 
                                               (torch.empty(0, dtype=torch.long, device=new_op_hidden.device),
                                                torch.empty(0, dtype=torch.long, device=new_op_hidden.device)))
                                                
        #if there are no sequence edges in the batch, return an empty sequence prediction
        if src_seq.size(0) == 0:
            out_seq = torch.empty((0, 1), device=new_op_hidden.device)
        else:
            ord_emb_i = static_embs['order'][src_seq]
            ord_emb_j = static_embs['order'][dst_seq]
            seq_edge_attr = edge_attr_dict[('order', 'to', 'order')]
            if seq_edge_attr.dim() == 1: seq_edge_attr = seq_edge_attr.unsqueeze(1)
            
            u_seq_edges = u[batch_dict['order'][src_seq]]
            
            assign_prob_matrix = torch.zeros((static_embs['order'].size(0), new_op_hidden.size(0)), device=new_op_hidden.device)
            assign_prob_matrix[dst_idx, src_idx] = out_assign.squeeze().detach()
            act_probs_1d = out_activation.squeeze().detach()
            
            chunk_size = 50000 
            shared_op_list, active_shared_list = [], []
            
            for start in range(0, src_seq.size(0), chunk_size):
                end = min(start + chunk_size, src_seq.size(0))
                probs_i = assign_prob_matrix[src_seq[start:end]]
                probs_j = assign_prob_matrix[dst_seq[start:end]]
                
                shared_op_list.append(torch.sum(probs_i * probs_j, dim=1, keepdim=True))
                active_shared_list.append(torch.sum(probs_i * probs_j * act_probs_1d, dim=1, keepdim=True))
            
            shared_op_score = self.head_coupling_dropout(torch.cat(shared_op_list, dim=0))
            active_shared_score = self.head_coupling_dropout(torch.cat(active_shared_list, dim=0))
            
            seq_input = torch.cat([ord_emb_i, ord_emb_j, u_seq_edges, seq_edge_attr, shared_op_score, active_shared_score], dim=1)
            out_seq = torch.sigmoid(self.seq_head(seq_input))

        #sequence masking
        valid_seq_mask = ((1.0 - is_assigned_flag[src_seq]) * (1.0 - is_assigned_flag[dst_seq]))
        out_seq = out_seq * valid_seq_mask

        return new_op_hidden, {
            'activation': out_activation, 
            'assignment': out_assign, 
            'sequence': out_seq 
        }

    def forward(self, x_dict_raw, edge_index_dict, edge_attr_dict, u, batch_dict=None, 
                static_embs=None, op_hidden=None, last_order_emb=None):
        """
        Runs the full RecGNN forward pass for a single sequential step.

        Args:
            x_dict_raw: The current dynamic node features (updated at each step).
            edge_index_dict, edge_attr_dict, u, batch_dict: Standard GNN inputs.
            static_embs: Cached embeddings from the Encoder. If None, the Encoder will run.
            op_hidden: The GRU hidden state from the previous step (t-1).
            last_order_emb: The embedding of the order completed in step (t-1).

        Returns:
            new_op_hidden: The updated GRU hidden state for step (t).
            predictions: Dict containing 'activation', 'assignment', and 'sequence' logits.
            static_embs: The cached encoder embeddings to pass into the next step.
        """
        if batch_dict is None:
            batch_dict = {key: torch.zeros(x.size(0), dtype=torch.long, device=x.device) for key, x in x_dict_raw.items()}

        #we use static encoder only if it hasn't been run yet (step 0)
        #so it is not re-running message passing in the loop.
        if static_embs is None:
            static_embs = self.encode(x_dict_raw, edge_index_dict, edge_attr_dict)

        #recurrent decoder step
        new_op_hidden, predictions = self.decode_step(
            static_embs=static_embs,
            x_dict_raw=x_dict_raw,
            edge_index_dict=edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            u=u,
            batch_dict=batch_dict,
            op_hidden=op_hidden, 
            last_order_emb=last_order_emb
        )
        
        #return the new recurrent state and the cached static embeddings for the next loop
        return new_op_hidden, predictions, static_embs
    
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




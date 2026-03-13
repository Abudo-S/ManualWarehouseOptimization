import math
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import HeteroConv, GATv2Conv, to_hetero
import os

SAVE_MODEL_PATH = "checkpoints/gnn_model_weights.pth"
SAVE_MODEL_IN_TRAINING_PATH = "checkpoints/gnn_checkpoint_epoch_idx.pth" #replace idx with epoch number when saving

class MultiCriteriaGNNModel(torch.nn.Module):
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
        '''
        super().__init__()

        assert heuristic_boost_factor >= 1.0,\
        "heuristic_boost_factor should be > 1.0"
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.head_coupling_dropout = torch.nn.Dropout(head_coupling_dropout)
        self.heuristic_boost_factor = heuristic_boost_factor

        #node encoders (project raw features to hidden dim)
        self.order_lin = Linear(10, hidden_dim) # mission features: 'WEIGHT', 'HEIGHT', 'WIDTH', 'LENGTH', 'FROM_X', 'FROM_Y', 'FROM_Z','TO_X', 'TO_Y', 'TO_Z'
        
        #physical features
        self.op_lin = Linear(7, hidden_dim) # operator features: 'SPEED', 'UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD', 'FORK_WIDTH', 'FORK_LENGTH'
        #self.op_lin = Linear(8, hidden_dim) # operator features: 'ID', 'SPEED', 'UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD', 'FORK_WIDTH', 'FORK_LENGTH'
        #self.op_lin = Linear(15, hidden_dim)  # 7 original + 8 positional encoding

        #message passing layers (encoder)
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            #a convolution per each edge type
            #note that we use edge_dim=1 because our time/processing features are 1D
            conv_dict = {
                ('order', 'to', 'order'): GATv2Conv(
                    hidden_dim, 
                    hidden_dim // heads, 
                    heads=heads, 
                    edge_dim=1, 
                    add_self_loops=False, 
                    dropout=dropout
                ),
                ('operator', 'assign', 'order'): GATv2Conv(
                    (hidden_dim, hidden_dim), 
                    hidden_dim // heads, 
                    heads=heads, 
                    edge_dim=2, #processing time + travel time from the base
                    add_self_loops=False, 
                    dropout=dropout
                ),

                #add reverse edges if the graph is bi-directional or needed for flow
                #operator updates (from potential orders)
                ('order', 'rev_assign', 'operator'): GATv2Conv(
                    (hidden_dim, hidden_dim), #(source=order, target=op)
                    hidden_dim // heads, 
                    heads=heads, 
                    edge_dim=2, #processing time + travel time from the base
                    add_self_loops=False, 
                    dropout=dropout
                )
            }
            
            # using aggr='mean' ensures node embeddings mathematically stay in a stable range, 
            # regardless of whether an operator processes 5 orders or 800+ orders.
            #HeteroConv wraps these standard GAT layers using aggregation function
            #using aggr='sum'. If an operator is connected to 10 missions, its feature values scale by ~10. 
            #If it is connected to 800 missions, its values scale by ~800. 
            #Across 3 GNN layers, this compounds exponentially ($800^3 = 5.12 \times 10^8$). 
            #Multiplying these massive numbers by network weights results in numeric overflow (inf -> NaN).
            #using aggr='mean': The node embeddings will mathematically stay in the [-1.0, 1.0] range,
            #regardless of whether an operator processes 5 orders or 800 orders.
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        #decision heads (decoders)
        
        #global context (u) has 3 dims: [Alpha, Beta, H_fixed]
        #we concat node_embedding (64) + global (3) = 67 inputs
        #+1 dim: [min_ops_needed]
        #input_dim_with_global = hidden_dim + 3
        input_dim_with_global = hidden_dim + 3 + 1 + 8
        
        #activation head (node classification for operators)
        self.activation_head = Sequential(
            Linear(input_dim_with_global, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1) #logits for binary classification
        )
        
        #assignment head (edge classification for op i -> order j)
        #input: op_embedding + order_embedding + global + edge_attr (time + base_travel_time) 
        #+ op_activation_prob (predicted by activation head) + raw_PE (8)
        #64 + 64 + 3 + 2 + 1 + 8 = 134
        input_dim_assignment  = 2 * hidden_dim + 6 + 8 #added 1 extra dim for op_activation_prob (activation coupling)
        self.assign_head = Sequential(
            #Linear(2 * hidden_dim + 3 + 1, hidden_dim), #decoupled head without activation feedback
            Linear(input_dim_assignment , hidden_dim), 
            ReLU(),
            Linear(hidden_dim, 1)
        )
        
        #sequence head (edge classification for order -> order)
        #input: order_embedding_i + order_embedding_j + global + edge_Attr (time) + shared_op_score (predicted by assignment head for both orders)
        #64 + 64 + 3 + 1 + 1 = 133
        #input: order_embedding_i + order_embedding_j + global + edge_Attr (time) + shared_op_score, active_shared_score
        #64 + 64 + 3 + 1 + 1 + 1 = 134
        input_dim_sequence = 2 * hidden_dim + 6 #explicit coupling to assignment & activation
        self.seq_head = Sequential(
            #Linear(2 * hidden_dim + 3 + 1, hidden_dim), #decoupled head without assignment feedback
            #Linear(2 * hidden_dim + 5, hidden_dim), #explicit coupling to assignment; therefore, implicit coupling to activation (though assignment)
            Linear(input_dim_sequence, hidden_dim), 
            ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, u, batch_dict=None):
        """
        The model performs message passing and then applies three separate heads for different classification tasks.
        x_dict: Node features {'order': [N, 7], 'operator': [M, 4]}
        edge_index_dict: Connectivity
        edge_attr_dict: Edge Features (Time) {'order__to__order': [E, 1], ...}
        u: Global params [Batch, 3]
        batch_dict: optional batching info for nodes
        Returns:
            Dict with keys:
            'activation': [num_ops, 1] logits for operator activation
            'assignment': [num_assign_edges, 1] logits for op->order assignment
            'sequence': [num_seq_edges, 1] logits for order->order sequencing
        Logits order is consistent with input order "edge_index".
        The logits are converted to probabilities via sigmoid.
        The probabilities can be interpreted as:
        - Activation: Probability that an operator should be activated.
        - Assignment: Probability that an operator should be assigned to a specific order.
        - Sequence: Probability that one order should precede another in the sequence.
        """
        #assuming operator features are [M, 15] where first 7 are physical, last 8 are PE
        raw_op_features = x_dict['operator']
        op_physical = raw_op_features[:, :7]
        op_pe = raw_op_features[:, 7:]

        #initial projection
        x_dict['order'] = self.order_lin(x_dict['order']).relu()
        #x_dict['operator'] = self.op_lin(x_dict['operator']).relu()
        x_dict['operator'] = self.op_lin(op_physical).relu() #pass only physical features through GNN
        
        #since positional encodings are being completely washed out by GATv2Conv (mean pairwise diff still 0.000000),
        #the message passing is averaging everything to death.
        #store original operator embeddings
        orig_op_emb = x_dict['operator'].clone()
        orig_ord_emb = x_dict['order'].clone()

        #message passing
        for conv in self.convs:
            #HeteroConv expects dicts. We pass edge attributes too.
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            
            #activation & residual could be added here
            #x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: torch.clamp(x, min=-1e4, max=1e4).relu() for key, x in x_dict.items()} #prevent exploding activations before ReLU
            x_dict = {key: torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4) for key, x in x_dict.items()} #prevent NaNs

        #mix original + message-passing embeddings
        #preserve identity while adding context
        x_dict['operator'] = 0.5 * orig_op_emb + 0.5 * x_dict['operator']
        x_dict['order'] = 0.5 * orig_ord_emb + 0.5 * x_dict['order']

        #if a single HeteroData object is passed (not a batch), batch_dict should be tensor of zeros
        if batch_dict is None:
            batch_dict = {
                key: torch.zeros(x.size(0), dtype=torch.long, device=x.device) 
                for key, x in x_dict.items()
            }

        #global context
        #u is [1, 3] (single graph batch). Broadcast to nodes if necessary or just concat.
        #an efficient way is to expand u to match node count during prediction.
        
        #head 1: activation (operator nodes)
        #expand u: [1, 3] -> [num_ops, 3]
        op_batch = batch_dict['operator']
        u_ops = u[op_batch] #match batch size for multiple graphs, shape: [num_ops, 3]

        #num_ops = x_dict['operator'].size(0)
        #u_ops = u.expand(num_ops, -1)
        
        #print((u_ops, x_dict['operator'])) #debugging before concat

        #STR- compute heuristic demand for activation head
        #assuming that batch_dict['order'] exists to group orders per graph in the batch
        assign_edges = edge_index_dict[('operator', 'assign', 'order')]
        proc_times = edge_attr_dict[('operator', 'assign', 'order')][:, 0]
        num_orders = x_dict['order'].size(0)
        num_graphs = u.size(0)
        
        #target node indices (orders)
        order_indices = assign_edges[1]
        
        sum_proc_per_order = torch.zeros(num_orders, dtype=proc_times.dtype, device=x_dict['order'].device)
        sum_proc_per_order.index_add_(0, order_indices, proc_times)
        
        count_per_order = torch.zeros(num_orders, dtype=proc_times.dtype, device=x_dict['order'].device)
        count_per_order.index_add_(0, order_indices, torch.ones_like(proc_times))
        
        #avoid division by zero for isolated nodes
        avg_proc_per_order = sum_proc_per_order / count_per_order.clamp(min=1.0)

        #sum the order workloads per graph (batch) using index_add_
        order_batch = batch_dict['order']
        total_workload_per_batch = torch.zeros(num_graphs, dtype=proc_times.dtype, device=x_dict['order'].device)
        total_workload_per_batch.index_add_(0, order_batch, avg_proc_per_order)

        #count operators per graph
        op_batch = batch_dict['operator']
        ops_per_batch = torch.zeros(num_graphs, dtype=torch.float, device=x_dict['operator'].device)
        ops_per_batch = ops_per_batch.index_add(0, op_batch, torch.ones_like(op_batch, dtype=torch.float))

        #calculate min ops needed (total_workload / h_fixed)
        #u[:, 2] is h_fixed (normalized)
        h_fixed_norm = u[:, 2] 
        #add epsilon to prevent division by zero just in case
        min_ops_needed = total_workload_per_batch / (h_fixed_norm + 1e-6)
        
        #add a simple heuristic factor for travel time
        min_ops_needed = min_ops_needed * self.heuristic_boost_factor

        #normalize into a ratio
        load_ratio_per_batch = min_ops_needed / ops_per_batch.clamp(min=1.0)

        #broadcast back to operators
        #op_demand_feature = min_ops_needed[batch_dict['operator']] 
        op_demand_feature = load_ratio_per_batch[batch_dict['operator']] #normalized

        #concat: [op_emb, global, op_demand]
        #we'll need the raw logits or probabilities to feed to the next head
        #op_feat_final = torch.cat([x_dict['operator'], u_ops], dim=1)
        #op_feat_final = torch.cat([x_dict['operator'], u_ops, op_demand_feature.unsqueeze(1)], dim=1) #add minimum estimated ops
        op_feat_final = torch.cat([
            x_dict['operator'], 
            u_ops, 
            op_demand_feature.unsqueeze(1),
            op_pe #injects explicit id/location logic
        ], dim=1) 

        #apply sigmoid to squash raw logits to [0, 1] probability
        out_activation = torch.sigmoid(self.activation_head(op_feat_final))

        #head 2: assignment (op -> order edges)
        #we need to gather embeddings for source (op) and dest (order)
        src_idx, dst_idx = edge_index_dict[('operator', 'assign', 'order')]
        
        #gather embeddings
        op_emb = x_dict['operator'][src_idx] #source operators
        ord_emb = x_dict['order'][dst_idx] #dest orders
        edge_attr = edge_attr_dict[('operator', 'assign', 'order')] #processing time
        
        #expand global u to match number of edges
        # num_edges = src_idx.size(0)
        # u_edges = u.expand(num_edges, -1)
        edge_batch_indices = op_batch[src_idx] 
        u_edges = u[edge_batch_indices] #match batch size for multiple graphs, shape: [num_edges, 3]
        
        #STR- assignment-activation coupling
        #op_activation_prob = out_activation[src_idx].detach() #detach to avoid backpropagation from activation head
        op_activation_prob = out_activation[src_idx]
        
        #concat: [op, order, global, time, op_activation_prob]
        #assign_input = torch.cat([op_emb, ord_emb, u_edges, edge_attr, op_activation_prob], dim=1)
        assign_input = torch.cat([
            op_emb, 
            ord_emb, 
            u_edges, 
            edge_attr, 
            op_activation_prob,
            op_pe[src_idx] #allows MLP to distinguish perfectly symmetric operators
        ], dim=1)

        #the model considers h_fixed by directly concatenating it to the input vector of the final decision head (assign MLP).
        #Which allows the MLPs to learn a decision boundary that depends on h_fixed.
        #apply sigmoid to squash raw logits to [0, 1] probability
        out_assign = torch.sigmoid(self.assign_head(assign_input))

        #head 3: sequence (order -> order edges)
        src_idx, dst_idx = edge_index_dict[('order', 'to', 'order')]
        
        ord_emb_i = x_dict['order'][src_idx]
        ord_emb_j = x_dict['order'][dst_idx]
        edge_attr = edge_attr_dict[('order', 'to', 'order')] #travel Time
        
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        # num_edges = src_idx.size(0)
        # u_edges = u.expand(num_edges, -1)
        ord_batch = batch_dict['order']
        edge_batch_indices = ord_batch[src_idx] # use 'order' node batch
        u_edges = u[edge_batch_indices]

        #STR- sequence-assignment coupling
        #create a dense probability matrix [num_orders, num_ops]
        num_orders = x_dict['order'].size(0)
        num_ops = x_dict['operator'].size(0)
        assign_prob_matrix = torch.zeros((num_orders, num_ops), device=x_dict['order'].device)
        
        #fill the matrix with predicted assignment probabilities
        #assign_idx is [2, num_assign_edges]: [0] is op, [1] is order
        op_indices = edge_index_dict[('operator', 'assign', 'order')][0]
        ord_indices = edge_index_dict[('operator', 'assign', 'order')][1]
        
        #assign_prob_matrix[ord_indices, op_indices] = out_assign.squeeze().detach() #detach to avoid backpropagation from assignment head
        assign_prob_matrix[ord_indices, op_indices] = out_assign.squeeze()

        #get activation probs
        #act_probs_1d = out_activation.squeeze().detach() #detach to avoid backpropagation from activation head 
        act_probs_1d = out_activation.squeeze()

        #extract the probability vectors for source (i) and destination (j) orders
        probs_i = assign_prob_matrix[src_idx] #[num_seq_edges, num_ops]
        probs_j = assign_prob_matrix[dst_idx] #[num_seq_edges, num_ops]
        
        #compute the "shared operator score" 
        #sum of element-wise multiplication across ops
        #it yields a high value only if both orders have a high probability for the same operator.
        shared_op_score = torch.sum(probs_i * probs_j, dim=1, keepdim=True) #[num_seq_edges, 1]

        #compute active shared operator score (aActivation coupling) to avoid the possibile vanishing activation MLPS trough assignment head
        #we weight the assignment overlap by the activation probability of the operator
        active_shared_score = torch.sum(probs_i * probs_j * act_probs_1d, dim=1, keepdim=True)

        #apply head coupling dropout
        shared_op_score = self.head_coupling_dropout(shared_op_score)
        active_shared_score = self.head_coupling_dropout(active_shared_score)

        #concat: [ord_i, ord_j, global, time, shared_op_score]
        #implicit activation MLP signal should be passed through the assignment head; therefore, shared_op_score ()
        #seq_input = torch.cat([ord_emb_i, ord_emb_j, u_edges, edge_attr, shared_op_score, active_shared_score], dim=1)

        #concat: [ord_i, ord_j, global, time, shared_op_score]
        #explicit activation MLP signal passed directly in active_shared_score
        seq_input = torch.cat([ord_emb_i, ord_emb_j, u_edges, edge_attr, shared_op_score, active_shared_score], dim=1)

        #the model considers h_fixed by directly concatenating it to the input vector of the final decision head (seq MLP).
        #Which allows the MLPs to learn a decision boundary that depends on h_fixed.
        #apply sigmoid to squash raw logits to [0, 1] probability
        out_seq = torch.sigmoid(self.seq_head(seq_input))
        
        return {
            'activation': out_activation, #[num_ops, 1]
            'assignment': out_assign, #[num_assign_edges, 1]
            'sequence': out_seq #[num_seq_edges, 1]
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




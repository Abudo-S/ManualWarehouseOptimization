import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import HeteroConv, GATv2Conv, to_hetero
import os

SAVE_MODEL_PATH = "checkpoints/gnn_model_weights.pth"
SAVE_MODEL_IN_TRAINING_PATH = "checkpoints/gnn_checkpoint_epoch_idx.pth" #replace idx with epoch number when saving

class MultiCriteriaGNNModel(torch.nn.Module):
    def __init__(self, metadata, hidden_dim=64, num_layers=3, heads=4):
        '''
        metadata: Tuple of (node_types, edge_types) from the heterogeneous graph
        hidden_dim: Dimension of hidden embeddings
        num_layers: Number of GNN layers
        heads: Number of attention heads in GATv2
        defines a multi-criteria GNN model with three heads:
        1.Activation Head: Classifies operator nodes as active/inactive
        2.Assignment Head: Classifies edges from operators to orders
        3.Sequence Head: Classifies edges between orders
        '''
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        #node encoders (project raw features to hidden dim)
        self.order_lin = Linear(10, hidden_dim) # mission features: 'WEIGHT', 'HEIGHT', 'WIDTH', 'LENGTH', 'FROM_X', 'FROM_Y', 'FROM_Z','TO_X', 'TO_Y', 'TO_Z'
        self.op_lin = Linear(7, hidden_dim) # operator features: 'SPEED', 'UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD', 'FORK_WIDTH', 'FORK_LENGTH'
        
        #message passing layers (encoder)
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            #a convolution per each edge type
            #note that we use edge_dim=1 because our time/processing features are 1D
            conv_dict = {
                ('order', 'to', 'order'): GATv2Conv(
                    hidden_dim, hidden_dim // heads, 
                    heads=heads, edge_dim=1, add_self_loops=False
                ),
                ('operator', 'assign', 'order'): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim // heads, 
                    heads=heads, edge_dim=1, add_self_loops=False
                ),

                #add reverse edges if the graph is bi-directional or needed for flow
                #operator updates (from potential orders)
                ('order', 'rev_assign', 'operator'): GATv2Conv(
                    (hidden_dim, hidden_dim), #(source=order, target=op)
                    hidden_dim // heads, 
                    heads=heads, 
                    edge_dim=1, 
                    add_self_loops=False
                )
            }
            
            #HeteroConv wraps these standard GAT layers
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        #decision heads (decoders)
        
        #global context (u) has 3 dims: [Alpha, Beta, H_fixed]
        #we concat node_embedding (64) + global (3) = 67 inputs
        input_dim_with_global = hidden_dim + 3
        
        #activation head (node classification for operators)
        self.activation_head = Sequential(
            Linear(input_dim_with_global, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1) #logits for binary classification
        )
        
        #assignment head (edge classification for op i -> order j)
        #input: op_embedding + order_embedding + global + edge_attr (time)
        #64 + 64 + 3 + 1 = 132
        self.assign_head = Sequential(
            Linear(2 * hidden_dim + 3 + 1, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )
        
        #sequence head (edge classification for order -> order)
        #input: order_embedding_i + order_embedding_j + global + edge_Attr (time)
        self.seq_head = Sequential(
            Linear(2 * hidden_dim + 3 + 1, hidden_dim),
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
        
        #initial projection
        x_dict['order'] = self.order_lin(x_dict['order']).relu()
        x_dict['operator'] = self.op_lin(x_dict['operator']).relu()
        
        #message passing
        for conv in self.convs:
            #HeteroConv expects dicts. We pass edge attributes too.
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            
            #activation & residual could be added here
            x_dict = {key: x.relu() for key, x in x_dict.items()}

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
        
        #concat: [op_emb, global]
        op_feat_final = torch.cat([x_dict['operator'], u_ops], dim=1)
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
        
        #concat: [op, order, global, time]
        assign_input = torch.cat([op_emb, ord_emb, u_edges, edge_attr], dim=1)

        #apply sigmoid to squash raw logits to [0, 1] probability
        out_assign = torch.sigmoid(self.assign_head(assign_input))

        #head 3: sequence (order -> order edges)
        src_idx, dst_idx = edge_index_dict[('order', 'to', 'order')]
        
        ord_emb_i = x_dict['order'][src_idx]
        ord_emb_j = x_dict['order'][dst_idx]
        edge_attr = edge_attr_dict[('order', 'to', 'order')] #travel Time
        
        # num_edges = src_idx.size(0)
        # u_edges = u.expand(num_edges, -1)
        ord_batch = batch_dict['order']
        edge_batch_indices = ord_batch[src_idx] # use 'order' node batch
        u_edges = u[edge_batch_indices]

        
        seq_input = torch.cat([ord_emb_i, ord_emb_j, u_edges, edge_attr], dim=1)

        #apply sigmoid to squash raw logits to [0, 1] probability
        out_seq = torch.sigmoid(self.seq_head(seq_input))
        
        return {
            'activation': out_activation, #[num_ops, 1]
            'assignment': out_assign, #[num_assign_edges, 1]
            'sequence': out_seq #[num_seq_edges, 1]
        }
    
    def save_model(self, save_path=SAVE_MODEL_PATH):
        #create checkpoints directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        #save model weights
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path=SAVE_MODEL_PATH):
        #load the model weights
        self.load_state_dict(torch.load(save_path))
        self.eval()

    def save_model_in_training(self,
                               optimizer, 
                               current_epoch, 
                               current_loss, 
                               save_weights_path=SAVE_MODEL_PATH,
                               save_path=SAVE_MODEL_IN_TRAINING_PATH):
        #save weights
        self.save_model(save_weights_path)

        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
                               optimizer,
                               save_weights_path=SAVE_MODEL_PATH,
                               save_path=SAVE_MODEL_IN_TRAINING_PATH):
        #load weights
        self.load_model(save_weights_path)

        #load training checkpoint
        checkpoint = torch.load(save_path.replace("idx", str(current_epoch)))

        #retrieve states
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        self.train()

        return optimizer, start_epoch, loss




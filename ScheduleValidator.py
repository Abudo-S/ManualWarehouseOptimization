import networkx as nx
import torch
import numpy as np
import itertools
import json
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from ScheduleEvaluator import ScheduleEvaluator
from MultiCriteriaGNNModel import MultiCriteriaGNNModel
from GnnScheduleDataset import GnnScheduleDataset

LARGE_SCALE_BATCH_NAME = "Batch9000M" #Batch1000M
TARGET_MINI_BATCH_SIZE = 10 #number of missions per mini-batch
LARGE_SCALE_MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}.csv"
PREPROCESSED_BATCH_DIR = f"./preprocessed/{LARGE_SCALE_BATCH_NAME}/Batch{TARGET_MINI_BATCH_SIZE}M_idx.xlsx" #idx to be replaced cluster idx
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts10W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = f"./schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
BATCH_SIZE = 32 #nice to be equal to 32 or 64 since we have small mini-batch instances

#default threshold for binary classification accurcy like logistic regression after sigmoid
#need to be tuned if the classes are imbalanced (can be relevated from classification report / roc curve)
CLASSIFICATION_THRESHOLD = 0.05

class ScheduleValidator:
    def __init__(self, 
                 act_threshold=CLASSIFICATION_THRESHOLD, 
                 assign_threshold=CLASSIFICATION_THRESHOLD, 
                 seq_threshold=CLASSIFICATION_THRESHOLD):
        """
        Initializes the ScheduleValidator with the given batch of data.
        Args:
            batch: A PyG Batch object containing the graph data for the scheduling problem.
            act_threshold (float): Threshold for activation head.
            assign_threshold (float): Threshold for assignment head.
            seq_threshold (float): Threshold for sequence head.
        """
        self.act_threshold = act_threshold
        self.assign_threshold = assign_threshold
        self.seq_threshold = seq_threshold
    
    @torch.no_grad()
    def decode_assignment_one_per_order(self, batch, p_assign):
        """
        Assignment feasibility:
        Decodes the assignment head predictions to select at most one operator for each order.
         - batch: the input batch containing graph data
         - p_assign: predicted probabilities for assignment edges (shape: [num_assign_edges, 1])
        Returns:
         - chosen: boolean mask of shape [num_assign_edges] indicating which edges are selected for
        """

        #edges: operator -> order
        src, dst = batch.edge_index_dict[('operator','assign','order')]
        p = p_assign.view(-1)

        num_orders = batch['order'].num_nodes
        chosen = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)

        #[greedy] for each order j, pick the edge (i->j) with max probability
        #(if you we want to allow "unassigned", only choose if max_p >= thr)
        max_p = torch.full((num_orders,), -1e9, device=p.device)
        max_e = torch.full((num_orders,), -1, dtype=torch.long, device=p.device)

        for e in range(p.numel()):
            j = dst[e].item()
            if p[e] > max_p[j]:
                max_p[j] = p[e]
                max_e[j] = e

        valid_orders = (max_e >= 0)
        chosen[max_e[valid_orders]] = True
        
        return chosen  #boolean mask over assignment edges
    
    @torch.no_grad
    def check_activation_feasibility(self, batch, chosen_act_mask, chosen_assign_mask):
        """
        Activation feasibility:
        Checks consistency between activation head and assignment head.edges
        - batch: the input batch containing graph data
        - chosen_act_mask: boolean mask of shape [num_operator_nodes] indicating which operators are
            predicted active by the activation head.
        - chosen_assign_mask: boolean mask of shape [num_assign_edges] indicating which assignment edges are selected
        Returns:
            is_consistent (bool): True if strict rules are met.
            stats (dict): Counts of violations.
        """
        # 1. Get Assigned Operators
        # Find which operators have at least one assignment edge selected
        assign_edge_index = batch.edge_index_dict[('operator', 'assign', 'order')]
        src_ops = assign_edge_index[0] # Source Operator IDs for all edges
        
        # Filter only chosen edges
        active_edges_src = src_ops[chosen_assign_mask]
        
        # Get unique operators that have work
        ops_with_work = torch.unique(active_edges_src)
        
        # 2. Get Predicted Active Operators
        # Convert mask to indices
        ops_predicted_active = torch.nonzero(chosen_act_mask.view(-1)).squeeze()
        
        # 3. Check Rule 1: "Ghost Worker" (Working but not Active)
        # Every op in ops_with_work MUST be in ops_predicted_active
        # We use CPU sets for easy difference check
        set_work = set(ops_with_work.cpu().numpy())
        set_active = set(ops_predicted_active.cpu().numpy())
        
        ghost_workers = set_work - set_active # In Work but NOT in Active
        
        # 4. Check Rule 2: "Idle Active" (Active but no Work)
        # This is usually allowed but wasteful
        idle_active = set_active - set_work # In Active but NOT in Work
        
        is_feasible = (len(ghost_workers) == 0)
        
        return is_feasible, {
            "ghost_workers": len(ghost_workers), # CRITICAL violation
            "idle_active": len(idle_active)      # Warning (inefficiency)
        }

    @torch.no_grad
    def check_sequence_feasibility(self, batch, chosen_assign_mask, chosen_seq_mask):
        """
        Sequence feasibility:
        Checks if the chosen sequence edges form valid, acyclic paths consistent 
        with the chosen assignments.
        - batch: the input batch containing edge_index_dict
        - chosen_assign_mask: boolean mask of shape [num_assign_edges] indicating which assignment edges are selected
        - chosen_seq_mask: boolean mask of shape [num_seq_edges] indicating which sequence edges are selected
        Returns:
            is_feasible (bool): True if all checks pass.
            metrics (dict): Breakdown of violations.
        """
        
        #get the edges selected by the model (prob > assign_threshold & prob > seq_threshold)
        assign_edges = batch.edge_index_dict[('operator', 'assign', 'order')][:, chosen_assign_mask]
        seq_edges = batch.edge_index_dict[('order', 'to', 'order')][:, chosen_seq_mask]
        
        #convert to CPU for NetworkX processing (easier for graph logic)
        assign_edges = assign_edges.cpu().numpy()
        seq_edges = seq_edges.cpu().numpy()
        
        num_orders = batch['order'].num_nodes
        num_ops = batch['operator'].num_nodes
        
        #assignment map: orderId -> operatorId
        #if an order is unassigned or multi-assigned, we catch it here
        order_to_op = {}
        violations = {"multi_assign": 0, "cross_op_seq": 0, "cycles": 0, "branching": 0}
        
        for i in range(assign_edges.shape[1]):
            op, order = assign_edges[0, i], assign_edges[1, i]
            if order in order_to_op:
                violations["multi_assign"] += 1 #order assigned twice!
            order_to_op[order] = op

        #build sequence graph
        G_seq = nx.DiGraph()
        G_seq.add_nodes_from(range(num_orders))
        G_seq.add_edges_from(seq_edges.T)
        
        #check sequence constraints
        
        #branching factor (flow conservation)
        #max out-degree and in-degree must be <= 1
        for n in G_seq.nodes():
            if G_seq.out_degree(n) > 1: violations["branching"] += 1
            if G_seq.in_degree(n) > 1: violations["branching"] += 1
            
        #cross-operator eequencing
        #if A -> B, they must be done by same operator
        for u, v in G_seq.edges():
            op_u = order_to_op.get(u, -1) #-1 if unassigned
            op_v = order_to_op.get(v, -2)
            
            if op_u != op_v:
                if op_u == -1 or op_v == -1:
                    print(f"Warning: Sequence edge ({u}->{v}) involves unassigned order(s) (op_u={op_u}, op_v={op_v})")
                violations["cross_op_seq"] += 1
                
        #cycle detection
        try:
            cycles = list(nx.simple_cycles(G_seq))
            if len(cycles) > 0:
                violations["cycles"] = len(cycles)
        except:
            #fallback for very large graphs if simple_cycles is too slow
            if not nx.is_directed_acyclic_graph(G_seq):
                violations["cycles"] = 1

        #verdict
        total_violations = sum(violations.values())
        is_feasible = (total_violations == 0)
        
        return is_feasible, violations
    
    def resolve_sequence_conflicts(self, batch, seq_probs, chosen_seq_mask):
        """
        Post-processing step:
        1.removes cycles of length 2 (A<->B) by keeping the higher probability edge.
        2.enforces Max-1-Out degree locally (if A->B and A->C, keep best).
        
        Returns:
            cleaned_mask (torch.BoolTensor): Updated mask with conflicts removed.
        """
        #get edges and probs
        seq_edges = batch.edge_index_dict[('order', 'to', 'order')]
        
        #work with indices where mask is true
        chosen_indices = torch.nonzero(chosen_seq_mask.view(-1)).squeeze()
        
        #if nothing chosen, return empty
        if chosen_indices.numel() == 0:
            return chosen_seq_mask
            
        #build map of (u, v) -> (prob, edge_index)
        #iterate only the chosen edges
        edges_list = seq_edges[:, chosen_indices].cpu().detach().numpy()
        probs_list = seq_probs.view(-1)[chosen_indices].cpu().detach().numpy()
        
        #dict key: tuple (u, v), value: (probability, original_index)
        edge_map = {}
        for i, idx in enumerate(chosen_indices.tolist()):
            u, v = edges_list[0, i], edges_list[1, i]
            p = probs_list[i]
            edge_map[(u, v)] = (p, idx)
            
        #detect and resolve A <-> B conflicts
        indices_to_remove = set()
        
        for (u, v), (p_uv, idx_uv) in edge_map.items():

            #check if reverse exists
            if (v, u) in edge_map:
                p_vu, idx_vu = edge_map[(v, u)]
                
                #if we haven't already processed this pair
                if idx_vu not in indices_to_remove and idx_uv not in indices_to_remove:
                    if p_uv > p_vu:
                        indices_to_remove.add(idx_vu) #remove reverse
                    else:
                        indices_to_remove.add(idx_uv) #remove forward
                        
        #new aask
        cleaned_mask = chosen_seq_mask.clone()
        if len(indices_to_remove) > 0:
            #convert set to tensor list
            remove_tensor = torch.tensor(list(indices_to_remove), device=chosen_seq_mask.device)
            cleaned_mask[remove_tensor] = False
            
        return cleaned_mask

    def evaluate_full_feasibility(self, batch, out):
        """
        runs decoder -> feasibility checks -> returns verdict
        """
        
        #decode assignments (fix constraints greedily)
        chosen_assign = self.decode_assignment_one_per_order(batch, out['assignment'])
        print(f"Assignment Probabilities (sample): {out['assignment'].view(-1)[:10].cpu().detach().numpy()}")

        #decode sequences (just thresholding for now, hard to greedily fix without solver)
        chosen_seq = (out['sequence'].view(-1) > self.seq_threshold)
        chosen_seq = self.resolve_sequence_conflicts(batch, out['sequence'], chosen_seq)

        #feasibility checks
        #activation consistency
        act_ok, act_stats = self.check_activation_feasibility(batch, out['activation'], chosen_assign)
        
        #sequence consistency
        seq_ok, seq_stats = self.check_sequence_feasibility(batch, chosen_assign, chosen_seq)
        
        #assignment coverage (no unassigned orders)
        #since decoder enforces 1-per-order, we just check if any were dropped due to low prob
        total_orders = batch['order'].num_nodes
        assigned_orders = chosen_assign.sum().item() #assuming 1-to-1 decoder
        unassigned = total_orders - assigned_orders
        
        assign_ok = (unassigned == 0)
        
        #report
        is_valid = (act_ok and seq_ok and assign_ok)
        
        report = {
            "valid": is_valid,
            "act_ok": act_ok, "act_errs": act_stats,
            "seq_ok": seq_ok, "seq_errs": seq_stats,
            "assign_ok": assign_ok, "unassigned": unassigned,
            "masks": {"assign": chosen_assign, "seq": chosen_seq}  #masks for optimality gap calculation
        }
        
        return is_valid, report
    
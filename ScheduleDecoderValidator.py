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

class ScheduleDecoderValidator:
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
        #(if we want to allow "unassigned", only choose if max_p >= thr)
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
        #assigned operators have at least one assignment edge selected
        assign_edge_index = batch.edge_index_dict[('operator', 'assign', 'order')]
        src_ops = assign_edge_index[0] #source operator ids for all edges
        
        #only chosen edges
        active_edges_src = src_ops[chosen_assign_mask]
        
        #unique operators that have work
        ops_with_work = torch.unique(active_edges_src)
        
        #indices of predicted active operators
        ops_predicted_active = torch.nonzero(chosen_act_mask.view(-1)).squeeze()
        
        #ghost worker (working but not active)
        #every op in ops_with_work must be in ops_predicted_active
        set_work = set(ops_with_work.cpu().numpy())
        set_active = set(ops_predicted_active.cpu().numpy())
        
        ghost_workers = set_work - set_active
        
        #idle activate operators (active but no work)
        idle_active = set_active - set_work
        
        is_feasible = (len(ghost_workers) == 0) #in work but not ative = infeasible violation
        
        return is_feasible, {
            "ghost_workers": len(ghost_workers), #violation
            "idle_active": len(idle_active) #(inefficient activations, not strictly infeasible but good to track)
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
    
    def export_schedule_to_json(self, batch, report, filename="schedule.json"):
        """
        Exports a valid schedule to json.
        Requires the 'report' object from 'evaluate_full_feasibility'.
        """
        if not report["valid"]:
            print("Warning: exporting an invalid schedule!")

        #masks and indices
        chosen_assign = report["masks"]["assign"]
        chosen_seq = report["masks"]["seq"]
        
        #edges
        assign_idx = batch.edge_index_dict[('operator', 'assign', 'order')]
        seq_idx = batch.edge_index_dict[('order', 'to', 'order')]
        
        #filter chosen edges by masks
        assign_idxs = assign_idx[:, chosen_assign].cpu().detach().numpy()
        seq_idxs = seq_idx[:, chosen_seq].cpu().detach().detach().numpy()
        
        #map order -> operator
        order_to_op = {}
        for i in range(assign_idxs.shape[1]):
            op, order = int(assign_idxs[0, i]), int(assign_idxs[1, i])
            order_to_op[order] = op
            
        #map order -> next order (adjacency list for sequences)
        #dict is used, since valid schedule has max 1 out-degree
        next_order_map = {}
        for i in range(seq_idxs.shape[1]):
            u, v = int(seq_idxs[0, i]), int(seq_idxs[1, i])
            next_order_map[u] = v
            
        #group orders by op
        op_orders = {}
        for order, op in order_to_op.items():
            if op not in op_orders: op_orders[op] = set()
            op_orders[op].add(order)
            
        schedule_data = {
            "metadata": {
                "num_orders": int(batch['order'].num_nodes),
                "num_operators": int(batch['operator'].num_nodes),
                "valid": report["valid"],
                "schedule_id": batch.schedule_id,
            },
            "operators": []
        }
        
        all_mission_ids = batch['order'].global_id.cpu().detach().tolist()
        all_operator_ids = batch['operator'].global_id.cpu().detach().tolist()
        
        for op_id, orders in op_orders.items():
            #build mini-graph for this op
            targets = set()
            for u in orders:
                if u in next_order_map:
                    v = next_order_map[u]
                    if v in orders: #ensure consistency in sequencing (should be guaranteed by feasibility checks)
                        targets.add(v)
            
            starts = list(orders - targets)
            
            if not starts:
                #cycle case fallback
                best_start = list(orders)[0] 
            elif len(starts) == 1:
                best_start = starts[0]
            else:
                print(f"Multiple start nodes detected for operator {op_id}: {starts}. Choosing one arbitrarily.")
                best_start = starts[0]

            #sort routes (if multiple disconnected components, effectively multiple routes)
            routes = []
            if best_start is not None:
                route = [0]
                curr = best_start
                # infinite loop protection with visited set, since the model might have cycles due to model errors (should be caught in validation, but just in case)
                # while True: # we will break when no more next node or next node is not in same operator's orders
                #     route.append(curr)
                #     if curr in next_order_map and next_order_map[curr] in orders:
                #         curr = next_order_map[curr]
                #     else:
                #         break

                visited = set()
                while True:
                    #cycle detection
                    if curr in visited:
                        print(f"Cycle detected in export at node {curr}. Breaking.")
                        route.append(f"{curr} (CYCLE)") 
                        break
                        
                    visited.add(curr)
                    #route.append(curr) #append order node_index
                    route.append(all_mission_ids[curr])  #map order node_index -> CD_MISSION
                    
                    #move to next
                    if curr in next_order_map and next_order_map[curr] in orders:
                        curr = next_order_map[curr]
                    else:
                        break #end of chain
                    
                routes.append(route)
                
            schedule_data["operators"].append({
                "operator_id": all_operator_ids[op_id], #map operator node_index -> OPERATOR_ID
                "assigned_orders_count": len(orders),
                "routes": routes #list of possibile routes (e.g. [[1, 5, 2]])
            })
            
        #save in file
        with open(filename, 'w') as f:
            json.dump(schedule_data, f, indent=4)
        
        print(f"Schedule exported to {filename}")

if __name__ == "__main__":
    #init dataset
    dataset = GnnScheduleDataset(
        schedule_dir=SCHEDULE_DIR,
        mission_base_path=MISSION_BATCH_DIR,
        edge_base_path=MISSION_BATCH_TRAVEL_DIR,
        pallet_types_file_path=UDC_TYPES_DIR,
        fork_path=FORK_LIFTS_DIR
    )

    sample_data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #tuned hyperparameters 
    best_conf = {
        'batch_size': 32,
        'hidden_dim': 64,
        'heads': 4,
        'dropout': 0.1,
        'lr_trunk': 0.001,
        'lr_activation': 0.01,
        'lr_assignment': 0.0001,
        'lr_sequence': 0.0005
    }

    #tuned thresholds for feasibility validation 
    best_thresholds =  {
        'activation': 0.2209,
        'assignment': 0.0457,
        'sequence': 0.0918
    }

    model = MultiCriteriaGNNModel(
        metadata=sample_data.metadata(),
        hidden_dim=best_conf['hidden_dim'],
        num_layers=3,
        heads=best_conf['heads'],
        dropout=best_conf['dropout']
    ).to(device)
    
    model.load_model() #load pre-trained model weights

    totals = {"n": 0, "act_f1": 0.0, "assign_f1": 0.0, "seq_f1": 0.0,
              "feasible": 0, "gap_sum": 0.0}
    
    BATCH_SIZE = 1 #we want to evaluate one instance for coherency with mip solver schedules
    schedule_evaluator = ScheduleEvaluator(model=model,
                                        schedule_dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        act_threshold=best_thresholds['activation'],
                                        assign_threshold=best_thresholds['assignment'],
                                        seq_threshold=best_thresholds['sequence']
                                        )
    
    loader = DataLoader(schedule_evaluator.schedule_val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    scheduleValidator = ScheduleDecoderValidator(
        act_threshold=best_thresholds['activation'],
        assign_threshold=best_thresholds['assignment'],
        seq_threshold=best_thresholds['sequence']
    )
    
    print("Starting Schedule Validation - total batches:", len(loader))

    idx = 0
    for batch in loader:
        batch = batch.to(device)
        batch_dict = {'operator': batch['operator'].batch, 'order': batch['order'].batch}

        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.u, batch_dict=batch_dict)

        is_valid, report = scheduleValidator.evaluate_full_feasibility(batch, out)

        idx = idx + 1
        scheduleValidator.export_schedule_to_json(batch, report, filename=f"predicted_{batch.schedule_id}.json")
        
        #print(report)
        if is_valid:
            print(f"Schedule [{idx}] is Feasible!")
            #safe to calculate optimality gap using report["masks"]
            pass
        else:
            print(f"Schedule [{idx}] is NOT Feasible!")
            print(f"Activation Feasibility: {report['act_ok']} with stats {report['act_errs']}")

            if not report["seq_ok"]:
                print(f"Sequence Invalid: {report['seq_errs']}")
import networkx as nx
import torch
import numpy as np
import pandas as pd
import os
import math
import itertools
import time
import json
import logging
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from scipy.optimize import linear_sum_assignment
from torch_geometric.loader import DataLoader
from ScheduleEvaluator import ScheduleEvaluator
from MultiCriteriaGNNModel import MultiCriteriaGNNModel
from MultiCriteriaGNNModel_AutoRegressive import MultiCriteriaGNNModel_AutoRegressive
from GnnScheduleDataset import GnnScheduleDataset

LARGE_SCALE_BATCH_NAME = "Batch10000M" #Batch1000M, Batch9000M or Batch10000M
TARGET_MINI_BATCH_SIZE = 10 #number of missions per mini-batch
LARGE_BATCH_DIR = "./datasets/large-batch/batch/"
LARGE_BATCH_TRAVEL_DIR = "./datasets/large-batch/travel/"
MISSION_LARGE_BATCH_DIR = "./datasets/large-batch/batch/Batch_1_100M_distanced_A1.0_B1000.0_H90.csv"
LARGE_SCALE_MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}.csv"
PREPROCESSED_BATCH_DIR = f"./preprocessed/{LARGE_SCALE_BATCH_NAME}/Batch{TARGET_MINI_BATCH_SIZE}M_idx.xlsx" #idx to be replaced cluster idx
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
MISSION_LARGE_BATCH_TRAVEL_DIR = "./datasets/large-batch/travel/Batch_1_100M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts10W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = f"./schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
PREDICTED_SCHEDULE_DIR = f"./predicted_schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
PREDICTED_LARGE_SCHEDULE_DIR = f"./predicted_schedules/large-scale/batch/"
#used to compare model confidence in one-shot decoded predictions
#tail insertions are orders that have probs under fixed thresholds, need to refined.
MINI_BATCH_TAIL_INSERTIONS_DIR = f"./tail_insertions/validation_mini-batches.json"
LARGE_BATCH_TAIL_INSERTIONS_DIR = f"./tail_insertions/test_large-batches.json"

BATCH_SIZE = 32 #nice to be equal to 32 or 64 since we have small mini-batch instances
H_FIXED_EXCEED_TOLERANCE_MIN = 0.0 #allow schedules to tolerate H_fixed exceedance 
MAX_ITERATIONS_PER_ORDER = 10 #max attempts to find a feasible operator for an order based on assignment probs (in iterative repair)

#default threshold for binary classification accurcy like logistic regression after sigmoid
#need to be tuned if the classes are imbalanced (can be relevated from classification report / roc curve)
CLASSIFICATION_THRESHOLD = 0.05
TEMPERATURE_SCALING_FACTOR = 0.001

os.makedirs(os.path.dirname('logs/'), exist_ok=True)
logging.basicConfig(
    filename=f'logs/schedule_decoder_{int(time.time())}.log', 
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
def colored_background_str(r, g, b, text):
    return f'\033[48;2;{r};{g};{b}m{text}\033[0m'

class ScheduleDecoder:
    batch_tail_insertions = {}
    batch_tail_insertions_with_new_activations = {}
    
    def __init__(self, 
                 act_threshold=CLASSIFICATION_THRESHOLD, 
                 assign_threshold=CLASSIFICATION_THRESHOLD, 
                 seq_threshold=CLASSIFICATION_THRESHOLD,
                 predicted_schedule_dir=PREDICTED_SCHEDULE_DIR):
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
        self.predicted_schedule_dir = predicted_schedule_dir
    
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
        set_work = set(ops_with_work.view(-1).cpu().numpy())
        set_active = set(ops_predicted_active.view(-1).cpu().numpy())
        
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
        violations = {"multi_assign": 0, "cross_op_seq": 0, "cycles": 0}
        
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
            
        #cross-operator sequencing
        #if A -> B, they must be done by same operator
        # for u, v in G_seq.edges():
        #     op_u = order_to_op.get(u, -1) #-1 if unassigned
        #     op_v = order_to_op.get(v, -2)
            
        #     if op_u != op_v:
        #         if op_u == -1 or op_v == -1:
        #             print(f"Warning: Sequence edge ({u}->{v}) involves unassigned order(s) (op_u={op_u}, op_v={op_v})")
        #         violations["cross_op_seq"] += 1
                
        #cycle detection (removed inside export_schedule method)
        # try:
        #     cycles = list(nx.simple_cycles(G_seq))
        #     if len(cycles) > 0:
        #         violations["cycles"] = len(cycles)
        # except:
        #     #fallback for very large graphs if simple_cycles is too slow
        #     if not nx.is_directed_acyclic_graph(G_seq):
        #         violations["cycles"] = 1

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
        #chosen_indices = torch.nonzero(chosen_seq_mask.view(-1)).squeeze()
        chosen_indices = torch.nonzero(chosen_seq_mask.view(-1)).view(-1)
        
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
                        
        #new mask
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
        np.set_printoptions(suppress=True, precision=6)
        #decode assignments (fix constraints greedily)
        chosen_assign = self.decode_assignment_one_per_order(batch, out['assignment'])
        #print(f"Assignment Probabilities (sample): {out['assignment'].view(-1)[:10].cpu().detach().numpy()}")

        #decode sequences (just thresholding for now, hard to greedily fix without solver)
        chosen_seq = (out['sequence'].view(-1) > self.seq_threshold)
        chosen_seq = self.resolve_sequence_conflicts(batch, out['sequence'], chosen_seq)

        #decode activations (thresholding)
        chosen_act = (out['activation'].view(-1) > self.act_threshold)
        print(f"Activation Probabilities (sample): {out['activation'].view(-1).cpu().detach().numpy()}")
        logging.info(f"Activation Probabilities (sample): {out['activation'].view(-1).cpu().detach().numpy()}")
        
        #feasibility checks
        #activation consistency
        act_ok, act_stats = self.check_activation_feasibility(batch, chosen_act, chosen_assign)
        
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
            "masks": { #masks for schedule building
                "act": chosen_act,
                "assign": chosen_assign,
                "seq": chosen_seq
            }  
        }
        
        return is_valid, report
    
    @torch.no_grad()
    def check_horizon_constraint(self, batch, schedule_data, global_time_scale=1.0):
        """
        Validates if operator routes exceed the time horizon (H_fixed).
        Considers only a SINGLE BATCH (global H_fixed is the same for everyone).
        Assumes H in data.u is in MINUTES and schedule times are in SECONDS.
        - batch: The HeteroData batch object (contains data.u for H_fixed).
        - schedule_data: The dictionary structure produced by export_schedule_to_json.
        - global_time_scale: The factor to convert time units in batch.u to match schedule times (e.g., if batch.u is in minutes and schedule times are in seconds, this should be 60).
        """
        violations = []
        
        if not hasattr(batch, 'u') or batch.u is None:
            return True, []
            
        #batch.u shape is [1, 3]
        h_fixed_mins = float(batch.u[0, 2].item()) * global_time_scale #recovery of original time scale
        
        if h_fixed_mins < 60:
            print(f"Warning: H_fixed is very low (<{h_fixed_mins} mins). Check if time units are correct.")

        #check all operators against the h_fixed limit
        for op_data in schedule_data["operators"]:
            routes = op_data["routes"]
            
            for i, route in enumerate(routes):
                if not route: continue
                
                #total work duration is the finish time of the last step
                last_step = route[-1]
                total_time = last_step["finish_time"]
                
                if total_time > (h_fixed_mins + H_FIXED_EXCEED_TOLERANCE_MIN):
                    violations.append({
                        "operator_id": op_data["operator_id"],
                        "route_idx": i,
                        "duration": round(total_time, 2),
                        "limit": h_fixed_mins,
                        "excess": round(total_time - h_fixed_mins, 2)
                    })

        is_valid = (len(violations) == 0)

        return is_valid, violations

    @torch.no_grad()
    def export_schedule_to_json(self, batch, report, out, filename="schedule.json"):
        """
        Exports a valid schedule to json using cluster-first greedy decoding.
        Uses assignments from 'report' to cluster orders by operator.
        Within each cluster, greedily(high seq probs) navigates sequence edges using probabilities from 'out'.
        """
        if not report["valid"]:
            print("Warning: exporting an invalid schedule!")
        
        #assign & seq probs
        p_assign = out['assignment'].view(-1).cpu().numpy()
        p_seq = out['sequence'].view(-1).cpu().numpy()
        
        #edge indices (cpu for faster list processing)
        assign_idx = batch.edge_index_dict[('operator', 'assign', 'order')].cpu().numpy()
        seq_idx = batch.edge_index_dict[('order', 'to', 'order')].cpu().numpy()
        
        #use the pre-validated mask
        chosen_assign_mask = report["masks"]["assign"].cpu().numpy()
        
        #map: order_id -> (operator id, assignment prob)
        op_clusters = {} #op_id -> list of order ids
        order_assign_data = {} #order_id -> (op_id, prob)

        #iterate chosen assignments, chosen_assign_mask aligns with assign_idx columns
        chosen_indices = np.where(chosen_assign_mask)[0]
        
        for idx in chosen_indices:
            op = int(assign_idx[0, idx])
            order = int(assign_idx[1, idx])
            prob = float(p_assign[idx])
            
            if op not in op_clusters: op_clusters[op] = []
            op_clusters[op].append(order)
            
            order_assign_data[order] = (op, prob)
            
        #build weighted sequence adjacency for each order (for greedy routing)
        #we need quick access to: neighbors of u, sorted by prob desc
        seq_adj = {}
        for i in range(seq_idx.shape[1]):
            u, v = int(seq_idx[0, i]), int(seq_idx[1, i])
            prob = float(p_seq[i])
            
            if u not in seq_adj: seq_adj[u] = []
            seq_adj[u].append((v, prob))
            
        #sort neighbors for greedy picking
        for u in seq_adj:
            seq_adj[u].sort(key=lambda x: x[1], reverse=True)
            
        schedule_data = {
            "metadata": {
                "num_orders": int(batch['order'].num_nodes),
                "num_operators": int(batch['operator'].num_nodes),
                "valid": report["valid"],
                "schedule_id": getattr(batch, 'schedule_id', 'unknown'),
            },
            "operators": []
        }
        
        #global ids for export
        all_mission_ids = batch['order'].global_id.cpu().tolist()
        all_operator_ids = batch['operator'].global_id.cpu().tolist()

        for op_idx, orders in op_clusters.items():
            if not orders: continue
            
            orders_set = set(orders)
            real_op_id = all_operator_ids[op_idx]
            
            #heuristic: pick order with highest assignment prob
            #(logic: "best fit" for the operator is likely the anchor/first task)
            
            #sort orders by assignment prob desc
            sorted_orders = sorted(orders, key=lambda o: order_assign_data[o][1], reverse=True)
            
            #consume orders from the set as we route them
            unvisited = orders_set.copy()
            routes = []
            
            #while we have orders left to route
            while unvisited:
                #pick best start from remaining unvisited
                #intersection of sorted_orders and unvisited
                start_node = None
                for cand in sorted_orders:
                    if cand in unvisited:
                        start_node = cand
                        break
                
                if start_node is None: break #should not happen
                
                #greedy route generation
                route = []
                curr = start_node
                
                while True:
                    unvisited.discard(curr)
                    route.append(all_mission_ids[curr])
                    
                    #find best next step
                    #edge (curr -> next) exists & next is in 'unvisited' cluster
                    best_next = None
                    if curr in seq_adj:
                        for neighbor, prob in seq_adj[curr]:
                            if neighbor in unvisited:
                                best_next = neighbor
                                break #found best valid neighbor
                    
                    if best_next is not None:
                        curr = best_next
                    else:
                        #dead end in this cluster
                        break
                
                routes.append(route)
            
            #add operator section
            schedule_data["operators"].append({
                "operator_id": real_op_id,
                #"internal_idx": int(op_idx),
                "assigned_orders_count": len(orders),
                "routes": routes
            })
            
        #ensure output directory exists
        os.makedirs(os.path.dirname(self.predicted_schedule_dir), exist_ok=True)

        #save schedule
        with open(os.path.join(self.predicted_schedule_dir, filename), 'w') as f:
            json.dump(schedule_data, f, indent=4)
        
        print(f"Schedule exported with name: {filename}")

    @torch.no_grad()
    def export_schedule_with_timings(self, batch, report, out, filename="schedule.json"):
        """
        Exports a valid schedule to JSON with timing information.
        Format per step: {mission_id, start_time, finish_time, processing_duration, travel_duration, successor}
        """
        #in case of separated normalization, we need to recover original time values for processing and travel times
        # scale_proc = 1.0
        # if hasattr(batch['operator', 'assign', 'order'], 'max_val'):
        #     #take mean or first if batched
        #     scale_proc = batch['operator', 'assign', 'order'].max_val.mean().item()
            
        # scale_travel = 1.0
        # if hasattr(batch['order', 'to', 'order'], 'max_val'):
        #     #take mean or first if batched
        #     scale_travel = batch['order', 'to', 'order'].max_val.mean().item()
        
        global_time_scale = 1.0
        if hasattr(batch, 'global_scale_factor'):
            #take mean or first if batched
            global_time_scale = batch.global_scale_factor.mean().item()

        if not hasattr(batch, 'u') or batch.u is None:
            return True, []
            
        #batch.u shape is [1, 3]
        h_fixed_mins = float(batch.u[0, 2].item()) * global_time_scale #recovery of original time scale
        
        if not report["valid"]:
            print("Warning: exporting an invalid schedule!")

        #prepare assign & seq probs
        p_assign = out['assignment'].view(-1).cpu().numpy()
        p_seq = out['sequence'].view(-1).cpu().numpy()
        
        #edge indices (cpu for easier list processing)
        assign_idx = batch.edge_index_dict[('operator', 'assign', 'order')].cpu().numpy()
        seq_idx = batch.edge_index_dict[('order', 'to', 'order')].cpu().numpy()
        
        #processing time (op -> order): (op_idx, order_idx) -> p time
        proc_time_map = {}
        assign_attr = batch.edge_attr_dict[('operator', 'assign', 'order')].cpu().numpy()
        
        for i in range(assign_idx.shape[1]):
            op, order = int(assign_idx[0, i]), int(assign_idx[1, i])
            proc_time_map[(op, order)] = float(assign_attr[i][0])
            
        #travel time (Order -> Order): (u, v) -> p time
        travel_time_map = {}
        seq_attr = batch.edge_attr_dict[('order', 'to', 'order')].cpu().numpy()
        
        for i in range(seq_idx.shape[1]):
            u, v = int(seq_idx[0, i]), int(seq_idx[1, i])
            travel_time_map[(u, v)] = float(seq_attr[i][0])
            
        #use the pre-validated mask
        chosen_assign_mask = report["masks"]["assign"].cpu().numpy()
        
        #map: order_id -> (operator id, assignment prob)
        op_clusters = {} # op_id -> list of order ids
        order_assign_data = {} #order_id -> (op_id, prob)
        
        chosen_indices = np.where(chosen_assign_mask)[0]
        
        for idx in chosen_indices:
            op = int(assign_idx[0, idx])
            order = int(assign_idx[1, idx])
            prob = float(p_assign[idx])
            
            if op not in op_clusters: op_clusters[op] = []
            op_clusters[op].append(order)
            
            order_assign_data[order] = (op, prob)
            
        #build weighted sequence adjacency for each order (for greedy routing)
        seq_adj = {}
        for i in range(seq_idx.shape[1]):
            u, v = int(seq_idx[0, i]), int(seq_idx[1, i])
            prob = float(p_seq[i])
            
            if u not in seq_adj: seq_adj[u] = []
            seq_adj[u].append((v, prob))
            
        #sort neighbors for greedy picking
        for u in seq_adj:
            seq_adj[u].sort(key=lambda x: x[1], reverse=True)
            
        #reconstruct schedule with timings
        schedule_data = {
            "metadata": {
                "num_orders": int(batch['order'].num_nodes),
                "num_operators": int(batch['operator'].num_nodes),
                "valid": report["valid"],
                "schedule_id": getattr(batch, 'schedule_id', 'unknown'),
            },
            "operators": []
        }
        
        #global ids for export
        all_mission_ids = batch['order'].global_id.cpu().tolist()
        all_operator_ids = batch['operator'].global_id.cpu().tolist()

        for op_idx, orders in op_clusters.items():
            if not orders: continue
            
            orders_set = set(orders)
            real_op_id = all_operator_ids[op_idx]
            
            #heuristic: pick order with highest assignment prob as anchor
            sorted_orders = sorted(orders, key=lambda o: order_assign_data[o][1], reverse=True)
            
            #consume orders from the set as we route them
            unvisited = orders_set.copy()
            routes = []
            
            while unvisited:
                #pick best start from remaining unvisited
                start_node = None
                start_proc_time = 0.0

                for cand in sorted_orders:
                    if cand in unvisited:
                        #check feasibility for single node route (start time + processing <= H)
                        p_t = proc_time_map.get((op_idx, cand), 0.0) * global_time_scale
                        
                        #assuming start at t=0 + travel from Base (0 if ignored)
                        if p_t <= h_fixed_mins:
                            start_node = cand
                            start_proc_time = p_t
                            break
                
                if start_node is None: break 
                
                #greedy route generation with timings
                route_steps = []
                curr = start_node
                current_time = 0.0 #reset time for new route

                start_t = current_time 
                finish_t = start_t + start_proc_time
                current_time = finish_t
                
                #add initial step
                step = {
                    "mission_id": all_mission_ids[curr],
                    "start_time": round(start_t, 2),
                    "finish_time": round(finish_t, 2),
                    "processing_duration": round(start_proc_time, 2),
                    "travel_duration": 0.0,
                    "successor": None,
                    "_internal_idx": int(curr)
                }
                route_steps.append(step)
                unvisited.discard(curr)

                #extend Route
                while True:
                    best_next = None
                    best_next_metrics = None #(travel_t, proc_t, finish_t)
                    
                    #find best next step
                    #edge (curr -> next) exists & next is in 'unvisited' cluster
                    if curr in seq_adj:
                        for neighbor, prob in seq_adj[curr]:
                            if neighbor in unvisited:
                                #calculate p & t times for this potential next step
                                t_travel = travel_time_map.get((int(curr), int(neighbor)), 0.0) * global_time_scale
                                t_proc = proc_time_map.get((op_idx, int(neighbor)), 0.0) * global_time_scale

                                #finish time
                                next_finish = current_time + t_travel + t_proc

                                #horizon check
                                if next_finish <= h_fixed_mins:
                                    best_next = neighbor
                                    best_next_metrics = (t_travel, t_proc, next_finish)
                                    break #found best valid neighbor
                    
                    if best_next is not None:
                        #save metrics for this step
                        t_t, p_t, f_t = best_next_metrics
                        
                        #update previous step's successor
                        route_steps[-1]["successor"] = all_mission_ids[best_next]
                        
                        #new step
                        step = {
                            "mission_id": all_mission_ids[best_next],
                            "start_time": round(current_time + t_t, 2),
                            "finish_time": round(f_t, 2),
                            "processing_duration": round(p_t, 2),
                            "travel_duration": round(t_t, 2),
                            "successor": None,
                            "_internal_idx": int(best_next)
                        }
                        route_steps.append(step)
                        
                        #update state
                        curr = best_next
                        current_time = f_t
                        unvisited.discard(curr)
                    else: #no valid next step, end route here
                        break
                
                #clean temp node internal idx
                for s in route_steps: del s["_internal_idx"]
                
                routes.append(route_steps)
            
            #add operator section with timings
            schedule_data["operators"].append({
                "operator_id": real_op_id,
                #internal_idx": int(op_idx),
                "assigned_orders_count": sum(len(r) for r in routes),
                "routes": routes
            })
        
        h_valid, h_violations = self.check_horizon_constraint(batch, schedule_data, global_time_scale)
    
        schedule_data["metadata"]["horizon_valid"] = h_valid
        schedule_data["metadata"]["horizon_violations"] = h_violations
        
        if not h_valid:
            print(f"Warning: {len(h_violations)} routes exceed time horizon (H={batch.u[0,2].item()})")

        #ensure output directory exists
        os.makedirs(os.path.dirname(self.predicted_schedule_dir), exist_ok=True)
        
        #save schedule
        with open(os.path.join(self.predicted_schedule_dir, filename), 'w') as f:
            json.dump(schedule_data, f, indent=4)
        
        print(f"Schedule exported with name: {filename}")

    @torch.no_grad()
    def export_schedule_with_timings_v2(self, batch, report, out, filename="schedule.json"):
        """
        Exports a schedule using "iterative re-assignment with tie-breaking" based on assignment and sequence probs.
        Strategy:
        1.Group operators by assignment probability buckets (e.g., 0.9, 0.8...).
        2.Within a bucket, pick the operator with the MOST currently assigned orders (Tie-Breaker).
        3.Attempt to route. If it fits, keep it.
        4.If it doesn't fit (rejected), try the next operator in the same bucket.
        5.If bucket exhausted, move to next lower probability bucket.
        """

        global_time_scale = 1.0
        if hasattr(batch, 'global_scale_factor'):
            #take mean or first if batched
            global_time_scale = batch.global_scale_factor.mean().item()

        if not hasattr(batch, 'u') or batch.u is None:
            return True, []

        #batch.u shape is [1, 3]
        h_fixed_mins = float(batch.u[0, 2].item()) * global_time_scale #recovery of original time scale

        #prepare assign & seq probs
        p_assign = out['assignment'].view(-1).cpu().numpy()
        p_seq = out['sequence'].view(-1).cpu().numpy()
        
        #edge indices (cpu for easier list processing)
        assign_idx = batch.edge_index_dict[('operator', 'assign', 'order')].cpu().numpy()
        seq_idx = batch.edge_index_dict[('order', 'to', 'order')].cpu().numpy()
        
        #processing time (op -> order): (op_idx, order_idx) -> p time
        proc_time_map = {}
        assign_attr = batch.edge_attr_dict[('operator', 'assign', 'order')].cpu().numpy()
        for i in range(assign_idx.shape[1]):
            val = float(assign_attr[i][0]) if assign_attr.ndim > 1 else float(assign_attr[i])
            proc_time_map[(assign_idx[0,i], assign_idx[1,i])] = val * global_time_scale

        #travel time (order -> order): (u, v) -> t time
        travel_time_map = {}
        seq_attr = batch.edge_attr_dict[('order', 'to', 'order')].cpu().numpy()
        for i in range(seq_idx.shape[1]):
            val = float(seq_attr[i][0]) if seq_attr.ndim > 1 else float(seq_attr[i])
            travel_time_map[(seq_idx[0,i], seq_idx[1,i])] = val * global_time_scale

        #build weighted sequence adjacency for each order (for greedy routing)
        seq_adj = {}
        for i in range(seq_idx.shape[1]):
            u, v = int(seq_idx[0, i]), int(seq_idx[1, i])
            prob = float(p_seq[i])
            if u not in seq_adj: seq_adj[u] = []
            seq_adj[u].append((v, prob))
        
        #sort neighbors for greedy picking
        for u in seq_adj:
            seq_adj[u].sort(key=lambda x: x[1], reverse=True)

        all_mission_ids = batch['order'].global_id.cpu().tolist()
        all_operator_ids = batch['operator'].global_id.cpu().tolist()
        num_orders = batch['order'].num_nodes
        
        #build the set of active operators from the activation mask
        chosen_act_mask = report["masks"]["act"].cpu().numpy()  #bool array [num_ops]
        active_ops = set(np.where(chosen_act_mask)[0])

        #if no operators pass the threshold, use top-k
        if not active_ops:
            print("Warning: no active operators from activation mask, using top-k fallback.")
            act_probs = out['activation'].view(-1).cpu().numpy()
            k = max(1, int(batch['operator'].num_nodes * 0.3))
            active_ops = set(np.argsort(act_probs)[-k:].tolist())

        #preference groups per order
        #map: order_id -> list of dicts [{'prob': 0.9, 'ops': [1, 5]}, ...]
        order_preferences = {}
        temp_prefs = {o: {} for o in range(num_orders)} #initialize for all orders
        

        for i in range(p_assign.shape[0]):
            op = int(assign_idx[0, i])
            order = int(assign_idx[1, i])
            prob = float(p_assign[i])
            
            if prob > 0.001: 
                #to its probability. This ensures active ops are always placed in 
                #buckets that are sorted before inactive ops.
                priority_boost = 10.0 if op in active_ops else 0.0
                effective_prob = prob + priority_boost

                #round to 4 decimals to detect "ties" effectively
                #(e.g., 0.9123 and 0.9124 might be treated as different buckets without rounding)
                prob_key = round(effective_prob, 4) 
                
                if prob_key not in temp_prefs[order]: 
                    temp_prefs[order][prob_key] = []
                temp_prefs[order][prob_key].append(op)
        
        #format and add fallback for orders with no valid probs
        all_ops_list = list(range(int(batch['operator'].num_nodes)))

        for o in range(num_orders):
            groups = temp_prefs[o]
            
            #if an order has no operators with prob > 0.001, we must provide a fallback
            #otherwise it immediately becomes a 'dead_order' and is left unassigned.
            if not groups:
                order_preferences[o] = [{'prob': 0.0, 'ops': all_ops_list}]
                continue

            sorted_probs = sorted(groups.keys(), reverse=True)
            order_preferences[o] = []
            for p in sorted_probs:
                orig_prob = p - 10.0 if p >= 10.0 else p
                order_preferences[o].append({'prob': orig_prob, 'ops': groups[p]})

            #filter out ops already present in previous buckets to avoid redundancy.
            seen_ops = set(op for b in order_preferences[o] for op in b['ops'])
            remaining_ops = [op for op in all_ops_list if op not in seen_ops]
            if remaining_ops:
                order_preferences[o].append({'prob': 0.0, 'ops': remaining_ops})

        #state tracking
        #order_bucket_idx[o] = k means trying the k-th probability bucket
        order_bucket_idx = {o: 0 for o in range(num_orders)}
        
        #order_tried_ops[o] = set() tracks specific ops we already failed with
        order_tried_ops = {o: set() for o in range(num_orders)}
        
        orders_to_assign = set(range(num_orders))
        final_op_routes = {} #op_id -> list of step dicts

        #iterative assignment loop
        MAX_ITERATIONS = num_orders * MAX_ITERATIONS_PER_ORDER 
        iteration = 0
        
        while orders_to_assign and iteration < MAX_ITERATIONS:
            iteration += 1
            
            #form vlusters
            current_clusters = {} #op -> set[orders]
            ops_to_process = set()
            
            #calculate current load of operators (for tie-breaking)
            #load = number of orders successfully routed in previous step
            op_loads = {op: len(route) for op, route in final_op_routes.items()}
            
            dead_orders = set()
            
            #assign pending orders to best available candidate
            for o in list(orders_to_assign):
                if o not in order_preferences:
                    dead_orders.add(o)
                    continue
                
                #find a valid bucket
                while True:
                    b_idx = order_bucket_idx[o]
                    if b_idx >= len(order_preferences[o]):
                        dead_orders.add(o) #exhausted all buckets
                        break
                    
                    bucket = order_preferences[o][b_idx]
                    candidates = bucket['ops']
                    
                    #filter candidates we haven't tried yet
                    valid_candidates = [op for op in candidates if op not in order_tried_ops[o]]
                    
                    if not valid_candidates:
                        #exhausted this bucket, move to next
                        order_bucket_idx[o] += 1
                        continue 
                    
                    #tie-breaking: pick candidate with highest current load (most assigned orders) to promote better clustering
                    #sort by: current load (desc), op_id (asc)
                    valid_candidates.sort(key=lambda op: (op_loads.get(op, 0), -op), reverse=True)
                    
                    #pick winner
                    best_op = valid_candidates[0]
                    
                    if best_op not in current_clusters: current_clusters[best_op] = set()
                    current_clusters[best_op].add(o)
                    ops_to_process.add(best_op)
                    
                    #mark as tentatively tried
                    order_tried_ops[o].add(best_op)
                    break
            
            for o in dead_orders:
                orders_to_assign.remove(o)

            #re-add existing routed orders
            #if an op is being re-processed, we must re-evaluate its entire load
            for op, route in final_op_routes.items():
                if op in ops_to_process:
                    existing = {s["_internal"] for s in route}
                    if op not in current_clusters: current_clusters[op] = set()
                    current_clusters[op].update(existing)

            #route rach cluster
            rejected_orders = set()
            
            for op in ops_to_process:
                orders = list(current_clusters[op])
                if not orders: continue
                
                #heuristic: sort by preference rank (try to keep "best fit" orders first)
                #we should use the probability of assignment to this op
                orders.sort(key=lambda o: next((grp['prob'] for grp in order_preferences.get(o, []) if op in grp['ops']), 0.0), reverse=True)
                
                unvisited = set(orders)
                route_steps = []
                current_time = 0.0
                curr = None
                
                #initial start (first order that fits)
                for cand in orders:
                    t_proc = proc_time_map.get((op, cand), 0.0)
                    if t_proc <= h_fixed_mins:
                        curr = cand
                        route_steps.append({
                            "mission_id": all_mission_ids[curr],
                            "_internal": curr,
                            "start_time": 0.0,
                            "finish_time": t_proc,
                            "processing_duration": t_proc,
                            "travel_duration": 0.0,
                            "successor": None
                        })
                        current_time = t_proc
                        unvisited.discard(curr)
                        break
                
                #extend route
                if curr is not None:
                    while True:
                        best_next = None
                        best_metrics = None
                        
                        if curr in seq_adj:
                            for neighbor, _ in seq_adj[curr]:
                                if neighbor in unvisited:
                                    t_travel = travel_time_map.get((curr, neighbor), 0.0)
                                    t_proc = proc_time_map.get((op, neighbor), 0.0)
                                    finish = current_time + t_travel + t_proc
                                    
                                    if finish <= h_fixed_mins:
                                        best_next = neighbor
                                        best_metrics = (t_travel, t_proc, finish)
                                        break
                        
                        if best_next:
                            t, p, f = best_metrics
                            route_steps[-1]["successor"] = all_mission_ids[best_next]
                            
                            route_steps.append({
                                "mission_id": all_mission_ids[best_next],
                                "_internal": best_next,
                                "start_time": round(current_time + t, 2),
                                "finish_time": round(f, 2),
                                "processing_duration": round(p, 2),
                                "travel_duration": round(t, 2),
                                "successor": None
                            })
                            current_time = f
                            curr = best_next
                            unvisited.discard(curr)
                        else:
                            break
                
                #commit valid route
                final_op_routes[op] = route_steps
                
                #register rejected orders (those in this cluster that were not assigned in the route)
                assigned_in_route = {s["_internal"] for s in route_steps}
                for o in orders:
                    if o not in assigned_in_route:
                        rejected_orders.add(o)

            #check convergence
            if not rejected_orders:
                break 
            
            #rejected orders go back to pool
            orders_to_assign = rejected_orders

        #reconstruct schedule with timings
        schedule_data = {
            "metadata": {
                "num_orders": int(batch['order'].num_nodes),
                "num_operators": int(batch['operator'].num_nodes),
                "valid": (len(final_op_routes) > 0), 
                "schedule_id": getattr(batch, 'schedule_id', 'unknown'),
            },
            "operators": []
        }

        assigned_count = 0
        for op_idx, route in final_op_routes.items():
            clean_route = [{k:v for k,v in s.items() if k!="_internal"} for s in route]
            if clean_route:
                schedule_data["operators"].append({
                    "operator_id": all_operator_ids[op_idx],
                    "assigned_orders_count": len(clean_route),
                    "routes": [clean_route]
                })
                assigned_count += len(clean_route)

        if assigned_count < num_orders:
            print(f"Warning: {num_orders - assigned_count} orders unassigned.")
            schedule_data["metadata"]["valid"] = False
        h_valid, h_violations = self.check_horizon_constraint(batch, schedule_data, global_time_scale)
    
        schedule_data["metadata"]["horizon_valid"] = h_valid
        schedule_data["metadata"]["horizon_violations"] = h_violations
        
        #ensure output directory exists
        os.makedirs(os.path.dirname(self.predicted_schedule_dir), exist_ok=True)
        
        #save schedule
        with open(os.path.join(self.predicted_schedule_dir, filename.replace('Batch', 'schedule')), 'w') as f:
            json.dump(schedule_data, f, indent=4)
        
        print(f"Schedule exported with name: {filename}")
    
    @torch.no_grad()
    def export_schedule_with_timings_v3(self, batch, out, filename="schedule.json", use_extra_ops=False, n_extra_ops_to_use=0):
        """
        Exports a schedule strictly following the auto-regressive logic: [activation -> assignment -> sequence]
        Activation: select active operators based on model probabilities.
        Assignment: greedily assign orders only to active operators based on assignment probs.
        Sequence: route assigned orders per operator using sequence probs and time constraints.
        If an order fails to route due to H_fixed, it falls back to the next best 
        activation-assignment probability bucket. If all active ops are full, it falls back to inactive ops.
        if use_extra_ops is True and there're unassigned order, 
        this mothod is repeated recursively trying to activate extra operators w.r.t. n_extra_ops_to_use.
        Tie-breaking logic steps:
        1.Activation prob (pack into the most confident active ops first)
        2.Raw assignment prob (best specific fit for this exact order)
        3.Current load (keep clusters together if probs are identical)
        4.Operator id (deterministic fallback)
        """
        global_time_scale = 1.0
        if hasattr(batch, 'global_scale_factor'):
            #take mean or first if batched
            global_time_scale = batch.global_scale_factor.mean().item()

        if not hasattr(batch, 'u') or batch.u is None:
            return True, []

        #batch.u shape is [1, 3]
        h_fixed_mins = float(batch.u[0, 2].item()) * global_time_scale

        #prepare probs of each head
        p_act = out['activation'].view(-1).cpu().numpy()
        p_assign = out['assignment'].view(-1).cpu().numpy()
        p_seq = out['sequence'].view(-1).cpu().numpy()

        #edge indices (cpu for easier list processing)
        assign_idx = batch.edge_index_dict[('operator', 'assign', 'order')].cpu().numpy()
        seq_idx = batch.edge_index_dict[('order', 'to', 'order')].cpu().numpy()

        #processing time and base travel time (op -> order)
        proc_time_map = {}
        base_travel_map = {}
        assign_attr = batch.edge_attr_dict[('operator', 'assign', 'order')].cpu().numpy()
        for i in range(assign_idx.shape[1]):
            val_proc = float(assign_attr[i][0]) if assign_attr.ndim > 1 else float(assign_attr[i])
            val_travel = float(assign_attr[i][1]) if assign_attr.ndim > 1 and assign_attr.shape[1] > 1 else 0.0
            
            op = int(assign_idx[0, i])
            order = int(assign_idx[1, i])
            
            proc_time_map[(op, order)] = val_proc * global_time_scale
            base_travel_map[(op, order)] = val_travel * global_time_scale

        travel_time_map = {}
        seq_attr = batch.edge_attr_dict[('order', 'to', 'order')].cpu().numpy()
        for i in range(seq_idx.shape[1]):
            val = float(seq_attr[i][0]) if seq_attr.ndim > 1 else float(seq_attr[i])
            #travel_time_map[(seq_idx[0, i], seq_idx[1, i])] = val * global_time_scale
            travel_time_map[(int(seq_idx[0, i]), int(seq_idx[1, i]))] = val * global_time_scale

        if travel_time_map:
            avg_travel_time = sum(travel_time_map.values()) / len(travel_time_map)
            if avg_travel_time == 0: #safety guard in case scaling is broken
                avg_travel_time = 1.0 #arbitrary fallback
        else:
            avg_travel_time = 1.0 #arbitrary fallback

        #travel time (order -> order): (u, v) -> t time weighted by sequence prob
        seq_adj = {}
        for i in range(seq_idx.shape[1]):
            u, v = int(seq_idx[0, i]), int(seq_idx[1, i])
            prob = float(p_seq[i])
            if u not in seq_adj: seq_adj[u] = []
            seq_adj[u].append((v, prob))
        for u in seq_adj:
            seq_adj[u].sort(key=lambda x: x[1], reverse=True) #sort by seq prob desc

        all_mission_ids = batch['order'].global_id.cpu().tolist()
        all_operator_ids = batch['operator'].global_id.cpu().tolist()
        num_orders = batch['order'].num_nodes
        num_ops = batch['operator'].num_nodes

        #STR- activation
        active_ops = set(np.where(p_act >= self.act_threshold)[0])

        if use_extra_ops:
            inactive_ops = [op for op in range(num_ops) if op not in active_ops]
            best_inactive_ops = sorted(inactive_ops, key=lambda x: p_act[x], reverse=True)[:n_extra_ops_to_use]
            
            for best_inactive_op in best_inactive_ops:
                active_ops.add(best_inactive_op)

        if not active_ops:
            k = max(1, int(num_ops * 0.3))
            active_ops = set(np.argsort(p_act)[-k:].tolist())

        total_processing_time = 0.0
        #estimate total workload (average processing time across all possible valid assignments)
        for o in range(num_orders):
            avg_proc = np.mean([proc_time_map.get((op, o), 0.0) for op in range(num_ops)])
            total_processing_time += avg_proc
            
        #STR- global Assignment

        #group candidates by strict probability buckets (quantized to 3 decimals)
        def get_bucket(prob): return round(float(prob), 3)

        #order_preferences[o] = dict mapping bucket -> list of valid operator candidates
        order_preferences = {o: {} for o in range(num_orders)}
        for i in range(p_assign.shape[0]):
            op = int(assign_idx[0, i])
            order = int(assign_idx[1, i])
            prob = float(p_assign[i])
            if prob > 0.001:
                b = get_bucket(prob)
                if b not in order_preferences[order]:
                    order_preferences[order][b] = []
                order_preferences[order][b].append(op)

        #order_bucket_state[o] tracks which sorted bucket index we are currently trying
        order_bucket_state = {o: 0 for o in range(num_orders)}
        #order_tried_ops[o] tracks specific operators already tried for order o
        order_tried_ops = {o: set() for o in range(num_orders)}
        
        #pre-sort buckets descending for each order
        order_sorted_buckets = {}
        for o in range(num_orders):
            order_sorted_buckets[o] = sorted(order_preferences[o].keys(), reverse=True)
            #fallback bucket for ops not explicitly predicted (prob 0.0)
            all_ops = list(range(num_ops))
            seen = set(op for ops in order_preferences[o].values() for op in ops)
            unseen = [op for op in all_ops if op not in seen]
            if unseen:
                order_preferences[o][0.0] = unseen
                order_sorted_buckets[o].append(0.0)

        orders_to_assign = set(range(num_orders))
        final_op_routes = {op: [] for op in range(num_ops)} #op -> list of steps

        MAX_ITERATIONS = num_orders * num_ops
        iteration = 0

        while orders_to_assign and iteration < MAX_ITERATIONS:
            iteration += 1

            #build temporary clusters for this assignment round
            current_clusters = {op: set() for op in range(num_ops)}
            ops_to_process = set()
            
            #current load (number of assigned orders per op) used as tie-breaker
            op_loads = {op: len(route) for op, route in final_op_routes.items()}
            
            dead_orders = set()

            for o in list(orders_to_assign):
                b_idx = order_bucket_state[o]
                found_candidate = False
                
                #scan down probability buckets until a valid candidate is found
                while b_idx < len(order_sorted_buckets[o]):
                    bucket_val = order_sorted_buckets[o][b_idx]
                    candidates = order_preferences[o][bucket_val]
                    
                    #filter untried candidates
                    valid_candidates = [c for c in candidates if c not in order_tried_ops[o]]
                    
                    if valid_candidates:
                        #hierarchical tie-breaker:
                        # 1. Is operator active? (True > False)
                        # 2. Activation Probability (higher is better)
                        # 3. Current Load (higher is better, keeps clusters dense)
                        # 4. Operator ID (deterministic fallback)
                        valid_candidates.sort(
                            key=lambda c: (c in active_ops, p_act[c], op_loads.get(c, 0), -c), 
                            reverse=True
                        )
                        best_op = valid_candidates[0]
                        current_clusters[best_op].add(o)
                        ops_to_process.add(best_op)
                        order_tried_ops[o].add(best_op)
                        found_candidate = True
                        break
                    else:
                        b_idx += 1
                        order_bucket_state[o] = b_idx
                
                if not found_candidate:
                    dead_orders.add(o)

            #remove completely exhausted orders (should rarely happen with fallback)
            for o in dead_orders:
                orders_to_assign.remove(o)

            #if no new assignments proposed, break to avoid infinite loop
            if not ops_to_process:
                break

            #add existing assigned orders to the clusters we are reprocessing
            for op in ops_to_process:
                existing = {s["_internal"] for s in final_op_routes[op]}
                current_clusters[op].update(existing)

            #STR- local Sequence Routing
            rejected_orders = set()

            for op in ops_to_process:
                orders = list(current_clusters[op])
                if not orders: continue
                
                #sort cluster by assignment preference to op
                #this prioritizes routing the orders that \"belong\" best to this operator
                def get_prob(o):
                    mask = (assign_idx[0] == op) & (assign_idx[1] == o)
                    idx_array = np.where(mask)[0]
                    return float(p_assign[idx_array[0]]) if len(idx_array) > 0 else 0.0

                orders.sort(key=lambda o: get_prob(o), reverse=True)

                unvisited = set(orders)
                route_steps = []
                current_time = 0.0
                curr = None

                #initial start: first valid task under time constraints
                for cand in orders:
                    t_travel = base_travel_map.get((op, cand), avg_travel_time) 
                    t_proc = proc_time_map.get((op, cand), 0.0)

                    if (t_proc + t_travel) <= h_fixed_mins:
                        curr = cand
                        route_steps.append({
                            "mission_id": all_mission_ids[curr],
                            "_internal": curr,
                            "start_time": round(t_travel, 2), #start after travel from base
                            "finish_time": round(t_travel + t_proc, 2),
                            "processing_duration": round(t_proc, 2),
                            "travel_duration": round(t_travel, 2),
                            "successor": None
                        })
                        current_time = t_proc + t_travel
                        unvisited.discard(curr)
                        break

                #route using sequence probs
                if curr is not None:
                    while unvisited:
                        best_next = None
                        best_metrics = None

                        #check greedy neighbors first
                        if curr in seq_adj:
                            for neighbor, _ in seq_adj[curr]:
                                if neighbor in unvisited:
                                    t_travel = travel_time_map.get((curr, neighbor), 0.0)
                                    t_proc = proc_time_map.get((op, neighbor), 0.0)
                                    finish = current_time + t_travel + t_proc

                                    if finish <= h_fixed_mins:
                                        best_next = neighbor
                                        best_metrics = (t_travel, t_proc, finish)
                                        break

                        #fallback if no neighbor fits
                        if not best_next:
                            for neighbor in unvisited:
                                t_travel = travel_time_map.get((curr, neighbor), avg_travel_time)
                                if t_travel == 0.0: t_travel = avg_travel_time 
                                t_proc = proc_time_map.get((op, neighbor), 0.0)
                                finish = current_time + t_travel + t_proc

                                if finish <= h_fixed_mins:
                                    best_next = neighbor
                                    best_metrics = (t_travel, t_proc, finish)
                                    break

                        if best_next:
                            t, p, f = best_metrics
                            route_steps[-1]["successor"] = all_mission_ids[best_next]
                            route_steps.append({
                                "mission_id": all_mission_ids[best_next],
                                "_internal": best_next,
                                "start_time": round(current_time + t, 2),
                                "finish_time": round(f, 2),
                                "processing_duration": round(p, 2),
                                "travel_duration": round(t, 2),
                                "successor": None
                            })
                            current_time = f
                            curr = best_next
                            unvisited.discard(curr)
                        else:
                            break #route full

                #commit route
                final_op_routes[op] = route_steps
                
                #track orders that couldn't be routed
                assigned_in_route = {s["_internal"] for s in route_steps}
                for o in orders:
                    if o not in assigned_in_route:
                        rejected_orders.add(o)

            orders_to_assign = rejected_orders

        if orders_to_assign:
            print(f"Tail-insertion pass: {len(orders_to_assign)} orders still unassigned")

            predicted_schedule_name = filename.replace(".json", "")
            if not predicted_schedule_name in self.batch_tail_insertions.keys():
                self.batch_tail_insertions[predicted_schedule_name] = num_orders - assigned_count

            for o in list(orders_to_assign):
                best_op = None
                best_finish = None
                best_t_travel = 0.0 #track the travel time for the chosen operator
                best_t_proc = 0.0  #track processing time for the winning op too
                    
                
                for op in active_ops:  #only try active operators
                    route = final_op_routes.get(op, [])
                    curr_finish = route[-1]["finish_time"] if route else 0.0
                    last_node = route[-1]['_internal'] if route else None
                    if last_node is not None:
                        t_travel = travel_time_map.get((last_node, o), avg_travel_time)
                    else:
                        t_travel = base_travel_map.get((op, o), avg_travel_time)
                    t_proc = proc_time_map.get((op, o), 0.0)
                    finish = curr_finish + t_travel + t_proc

                    if finish <= h_fixed_mins:
                        if best_finish is None or finish < best_finish:
                            best_finish = finish
                            best_op = op
                            best_t_travel = t_travel 
                            best_t_proc = t_proc 
                
                #if we found a spot, append it!
                if best_op is not None:
                    route = final_op_routes[best_op]
                    travel_start_time = route[-1]["finish_time"] if route else 0.0
                    step = {
                        "mission_id": all_mission_ids[o],
                        "_internal": o,
                        "start_time": round(travel_start_time + best_t_travel, 2), 
                        "finish_time": round(travel_start_time + best_t_travel + best_t_proc, 2), 
                        "processing_duration": round(best_t_proc, 2),
                        "travel_duration": round(best_t_travel, 2), 
                        "successor": None,
                    }
                    if route:
                        route[-1]["successor"] = all_mission_ids[o]
                    route.append(step)
                    orders_to_assign.remove(o)
                    print(f"Tail-inserted order {o} to Op {best_op}")


        #reconstruct output schedule
        schedule_data = {
            "metadata": {
                "num_orders": int(num_orders),
                "num_operators": int(num_ops),
                "valid": True,
                "schedule_id": getattr(batch, 'schedule_id', ['unknown'])[0] if hasattr(batch, 'schedule_id') else 'unknown',
            },
            "operators": []
        }

        assigned_count = 0
        for op_idx, route in final_op_routes.items():
            if route:
                clean_route = [{k:v for k,v in s.items() if k!="_internal"} for s in route]
                schedule_data["operators"].append({
                    "operator_id": all_operator_ids[op_idx],
                    "assigned_orders_count": len(clean_route),
                    "routes": [clean_route]
                })
                assigned_count += len(clean_route)

        activate_extra_op = False

        if assigned_count < num_orders and len(active_ops) < num_ops:
            print(f"Warning: {num_orders - assigned_count} orders unassigned.")
            logging.info(f"Warning: {num_orders - assigned_count} orders unassigned.")

            activate_extra_op = True

        predicted_schedule_name = filename.replace(".json", "")
        if not predicted_schedule_name in self.batch_tail_insertions_with_new_activations.keys():
            self.batch_tail_insertions_with_new_activations[predicted_schedule_name] = num_orders - assigned_count

        if activate_extra_op and len(active_ops) + n_extra_ops_to_use < np.size(p_act):
            n_extra_ops_to_use = n_extra_ops_to_use + 1
            print(f"Trying to resolve by activating extra {n_extra_ops_to_use} operators.")
            logging.info(f"Trying to resolve by activating extra {n_extra_ops_to_use} operators.")

            self.export_schedule_with_timings_v3(batch=batch, out=out, filename=filename, use_extra_ops=True, n_extra_ops_to_use=n_extra_ops_to_use)
        else:
            unassigned_orders = num_orders - assigned_count
            schedule_data["metadata"]["unassigned_orders"] = unassigned_orders
            if unassigned_orders > 0:
                print(colored_background_str(r=255, g=0, b=5, text=f"Warning: {unassigned_orders} orders remain unassigned."))

            h_valid, h_violations = self.check_horizon_constraint(batch, schedule_data, global_time_scale)

            schedule_data["metadata"]["horizon_valid"] = h_valid
            schedule_data["metadata"]["horizon_violations"] = h_violations

            #ensure output directory exists
            os.makedirs(os.path.dirname(self.predicted_schedule_dir), exist_ok=True)

            #save schedule
            with open(os.path.join(self.predicted_schedule_dir, filename.replace('Batch', 'schedule')), 'w') as f:
                json.dump(schedule_data, f, indent=4)

            print(f"Schedule exported with name: {filename}")
            logging.info(f"Schedule exported with name: {filename}")


    @torch.no_grad()
    def export_schedule_with_timings_v3_hungarian(self, batch, out, filename="schedule.json", use_extra_ops=False, n_extra_ops_to_use=0):
        """
        Exports a schedule strictly following the auto-regressive logic: [activation -> assignment -> sequence]
        Uses the Hungarian Algorithm (linear_sum_assignment) for global fairness distribution 
        instead of greedy probability buckets.
        """
        global_time_scale = 1.0
        if hasattr(batch, 'global_scale_factor'):
            #take mean or first if batched
            global_time_scale = batch.global_scale_factor.mean().item()

        if not hasattr(batch, 'u') or batch.u is None:
            return True, []

        #batch.u shape is [1, 3]
        h_fixed_mins = float(batch.u[0, 2].item()) * global_time_scale

        #prepare probs of each head
        p_act = out['activation'].view(-1).cpu().numpy()
        p_assign = out['assignment'].view(-1).cpu().numpy()
        p_seq = out['sequence'].view(-1).cpu().numpy()

        #edge indices (cpu for easier list processing)
        assign_idx = batch.edge_index_dict[('operator', 'assign', 'order')].cpu().numpy()
        seq_idx = batch.edge_index_dict[('order', 'to', 'order')].cpu().numpy()

        #processing time and base travel time (op -> order)
        proc_time_map = {}
        base_travel_map = {}
        assign_attr = batch.edge_attr_dict[('operator', 'assign', 'order')].cpu().numpy()
        for i in range(assign_idx.shape[1]):
            val_proc = float(assign_attr[i][0]) if assign_attr.ndim > 1 else float(assign_attr[i])
            val_travel = float(assign_attr[i][1]) if assign_attr.ndim > 1 and assign_attr.shape[1] > 1 else 0.0
            
            op = int(assign_idx[0, i])
            order = int(assign_idx[1, i])
            
            proc_time_map[(op, order)] = val_proc * global_time_scale
            base_travel_map[(op, order)] = val_travel * global_time_scale

        travel_time_map = {}
        seq_attr = batch.edge_attr_dict[('order', 'to', 'order')].cpu().numpy()
        for i in range(seq_idx.shape[1]):
            val = float(seq_attr[i][0]) if seq_attr.ndim > 1 else float(seq_attr[i])
            travel_time_map[(int(seq_idx[0, i]), int(seq_idx[1, i]))] = val * global_time_scale

        if travel_time_map:
            avg_travel_time = sum(travel_time_map.values()) / len(travel_time_map)
            if avg_travel_time == 0: avg_travel_time = 1.0 
        else:
            avg_travel_time = 1.0 

        #travel time (order -> order): (u, v) -> prob
        seq_adj = {}
        for i in range(seq_idx.shape[1]):
            u, v = int(seq_idx[0, i]), int(seq_idx[1, i])
            prob = float(p_seq[i])
            if u not in seq_adj: seq_adj[u] = []
            seq_adj[u].append((v, prob))
        for u in seq_adj:
            seq_adj[u].sort(key=lambda x: x[1], reverse=True) 

        all_mission_ids = batch['order'].global_id.cpu().tolist()
        all_operator_ids = batch['operator'].global_id.cpu().tolist()
        num_orders = batch['order'].num_nodes
        num_ops = batch['operator'].num_nodes

        #STR- activation
        active_ops = set(np.where(p_act >= self.act_threshold)[0])

        if use_extra_ops:
            inactive_ops = [op for op in range(num_ops) if op not in active_ops]
            best_inactive_ops = sorted(inactive_ops, key=lambda x: p_act[x], reverse=True)[:n_extra_ops_to_use]
            
            for best_inactive_op in best_inactive_ops:
                active_ops.add(best_inactive_op)

        if not active_ops:
            k = max(1, int(num_ops * 0.3))
            active_ops = set(np.argsort(p_act)[-k:].tolist())

        #STR- hungarian global assignment
        num_active = max(1, len(active_ops))
        #estimate max capacity: assume each operator can at most take (total_orders / active_ops) * margin
        max_capacity_per_op = int(math.ceil((num_orders / num_active) * 3.0))

        active_ops_list = list(active_ops)
        total_slots = len(active_ops_list) * max_capacity_per_op

        #initialize to 10.0 (high penalty because we are minimizing cost)
        cost_matrix = np.full((num_orders, total_slots), 10.0) 

        #map slot column index back to operator id
        col_to_op = {}
        col_idx = 0
        for op in active_ops_list:
            for _ in range(max_capacity_per_op):
                col_to_op[col_idx] = op
                col_idx += 1

        #fill Cost Matrix
        for o in range(num_orders):
            for col in range(total_slots):
                op = col_to_op[col]
                #find edge probability
                mask = (assign_idx[0] == op) & (assign_idx[1] == o)
                idx_array = np.where(mask)[0]

                if len(idx_array) > 0:
                    prob = float(p_assign[idx_array[0]])
                    #only consider valid assignments
                    if prob > 0.0:
                        #cost is negative probability (maximizing prob = minimizing cost)
                        cost_matrix[o, col] = -prob 

        #solve Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        #build clusters from matches
        current_clusters = {op: set() for op in range(num_ops)}
        ops_to_process = set()
        orders_to_assign = set(range(num_orders))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0: #only accept if it was a valid probability match
                assigned_op = col_to_op[c]
                current_clusters[assigned_op].add(r)
                ops_to_process.add(assigned_op)
                orders_to_assign.remove(r)

        #STR- local sequence routing
        final_op_routes = {op: [] for op in range(num_ops)} 
        rejected_orders = set(orders_to_assign) #start with orders Hungarian couldn't place

        for op in ops_to_process:
            orders = list(current_clusters[op])
            if not orders: continue
            
            #sort cluster by assignment preference to op
            def get_prob(o):
                mask = (assign_idx[0] == op) & (assign_idx[1] == o)
                idx_array = np.where(mask)[0]
                return float(p_assign[idx_array[0]]) if len(idx_array) > 0 else 0.0

            orders.sort(key=lambda o: get_prob(o), reverse=True)

            unvisited = set(orders)
            route_steps = []
            current_time = 0.0
            curr = None

            #initial start
            for cand in orders:
                t_travel = base_travel_map.get((op, cand), avg_travel_time) 
                t_proc = proc_time_map.get((op, cand), 0.0)

                if (t_proc + t_travel) <= h_fixed_mins:
                    curr = cand
                    route_steps.append({
                        "mission_id": all_mission_ids[curr],
                        "_internal": curr,
                        "start_time": round(t_travel, 2),
                        "finish_time": round(t_travel + t_proc, 2),
                        "processing_duration": round(t_proc, 2),
                        "travel_duration": round(t_travel, 2),
                        "successor": None
                    })
                    current_time = t_proc + t_travel
                    unvisited.discard(curr)
                    break

            #route using sequence probs
            if curr is not None:
                while unvisited:
                    best_next = None
                    best_metrics = None

                    if curr in seq_adj:
                        for neighbor, _ in seq_adj[curr]:
                            if neighbor in unvisited:
                                t_travel = travel_time_map.get((curr, neighbor), 0.0)
                                t_proc = proc_time_map.get((op, neighbor), 0.0)
                                finish = current_time + t_travel + t_proc

                                if finish <= h_fixed_mins:
                                    best_next = neighbor
                                    best_metrics = (t_travel, t_proc, finish)
                                    break

                    if not best_next:
                        for neighbor in unvisited:
                            t_travel = travel_time_map.get((curr, neighbor), avg_travel_time)
                            if t_travel == 0.0: t_travel = avg_travel_time 
                            t_proc = proc_time_map.get((op, neighbor), 0.0)
                            finish = current_time + t_travel + t_proc

                            if finish <= h_fixed_mins:
                                best_next = neighbor
                                best_metrics = (t_travel, t_proc, finish)
                                break

                    if best_next:
                        t, p, f = best_metrics
                        route_steps[-1]["successor"] = all_mission_ids[best_next]
                        route_steps.append({
                            "mission_id": all_mission_ids[best_next],
                            "_internal": best_next,
                            "start_time": round(current_time + t, 2),
                            "finish_time": round(f, 2),
                            "processing_duration": round(p, 2),
                            "travel_duration": round(t, 2),
                            "successor": None
                        })
                        current_time = f
                        curr = best_next
                        unvisited.discard(curr)
                    else:
                        break

            final_op_routes[op] = route_steps
            
            assigned_in_route = {s["_internal"] for s in route_steps}
            for o in orders:
                if o not in assigned_in_route:
                    rejected_orders.add(o)

        #STR- tail insertion for rejected/unassigned orders
        if rejected_orders:
            print(f"Tail-insertion pass: {len(rejected_orders)} orders still unassigned")
            logging.info(f"Tail-insertion pass: {len(rejected_orders)} orders still unassigned")
            
            for o in list(rejected_orders):
                best_op = None
                best_finish = None
                best_t_travel = 0.0 
                best_t_proc = 0.0  
                
                for op in active_ops: 
                    route = final_op_routes.get(op, [])
                    curr_finish = route[-1]["finish_time"] if route else 0.0
                    last_node = route[-1]['_internal'] if route else None
                    if last_node is not None:
                        t_travel = travel_time_map.get((last_node, o), avg_travel_time)
                    else:
                        t_travel = base_travel_map.get((op, o), avg_travel_time)
                    t_proc = proc_time_map.get((op, o), 0.0)
                    finish = curr_finish + t_travel + t_proc

                    if finish <= h_fixed_mins:
                        if best_finish is None or finish < best_finish:
                            best_finish = finish
                            best_op = op
                            best_t_travel = t_travel 
                            best_t_proc = t_proc 
                
                if best_op is not None:
                    route = final_op_routes[best_op]
                    travel_start_time = route[-1]["finish_time"] if route else 0.0
                    step = {
                        "mission_id": all_mission_ids[o],
                        "_internal": o,
                        "start_time": round(travel_start_time + best_t_travel, 2), 
                        "finish_time": round(travel_start_time + best_t_travel + best_t_proc, 2), 
                        "processing_duration": round(best_t_proc, 2),
                        "travel_duration": round(best_t_travel, 2), 
                        "successor": None,
                    }
                    if route:
                        route[-1]["successor"] = all_mission_ids[o]
                    route.append(step)
                    rejected_orders.remove(o)
                    print(f"Tail-inserted order {o} to Op {best_op}")
                    logging.info(f"Tail-inserted order {o} to Op {best_op}")


        #reconstruct output schedule
        schedule_data = {
            "metadata": {
                "num_orders": int(num_orders),
                "num_operators": int(num_ops),
                "valid": True,
                "schedule_id": getattr(batch, 'schedule_id', ['unknown'])[0] if hasattr(batch, 'schedule_id') else 'unknown',
            },
            "operators": []
        }

        assigned_count = 0
        for op_idx, route in final_op_routes.items():
            if route:
                clean_route = [{k:v for k,v in s.items() if k!="_internal"} for s in route]
                schedule_data["operators"].append({
                    "operator_id": all_operator_ids[op_idx],
                    "assigned_orders_count": len(clean_route),
                    "routes": [clean_route]
                })
                assigned_count += len(clean_route)

        activate_extra_op = False
        if assigned_count < num_orders and len(active_ops) < num_ops:
            print(f"Warning: {num_orders - assigned_count} orders unassigned.")
            logging.critical(f"Warning: {num_orders - assigned_count} orders unassigned.")

            activate_extra_op = True

        predicted_schedule_name = filename.replace(".json", "")
        if not predicted_schedule_name in self.batch_tail_insertions.keys():
            self.batch_tail_insertions[predicted_schedule_name] = num_orders - assigned_count

        if activate_extra_op and len(active_ops) + n_extra_ops_to_use < np.size(p_act):
            n_extra_ops_to_use = n_extra_ops_to_use + 1
            print(f"Trying to resolve by activating extra {n_extra_ops_to_use} operators.")
            logging.info(f"Trying to resolve by activating extra {n_extra_ops_to_use} operators.")

            self.export_schedule_with_timings_v3_hungarian(batch=batch, out=out, filename=filename, use_extra_ops=True, n_extra_ops_to_use=n_extra_ops_to_use)
        else:
            unassigned_orders = num_orders - assigned_count
            schedule_data["metadata"]["unassigned_orders"] = unassigned_orders
            if unassigned_orders > 0:
                print(colored_background_str(r=255, g=0, b=5, text=f"Warning: {unassigned_orders} orders remain unassigned."))

            h_valid, h_violations = self.check_horizon_constraint(batch, schedule_data, global_time_scale)

            schedule_data["metadata"]["horizon_valid"] = h_valid
            schedule_data["metadata"]["horizon_violations"] = h_violations

            #ensure output directory exists
            os.makedirs(os.path.dirname(self.predicted_schedule_dir), exist_ok=True)

            #save schedule
            with open(os.path.join(self.predicted_schedule_dir, filename.replace('Batch', 'schedule')), 'w') as f:
                json.dump(schedule_data, f, indent=4)

            print(f"Schedule exported with name: {filename}")
            logging.info(f"Schedule exported with name: {filename}")


    @torch.no_grad()
    def export_schedule_with_timings_v3_hungarian_seq_refined(self, batch, out, filename="schedule.json", use_extra_ops=False, n_extra_ops_to_use=0):
        """
        Exports a schedule strictly following the auto-regressive logic: [activation -> assignment -> sequence]
        Uses the Hungarian Algorithm (linear_sum_assignment) for global fairness distribution.
        Includes a completely non-destructive Makespan Minimization Refinement pass.
        """
        # if '800' in filename:
        #     use_extra_ops = True
        #     n_extra_ops_to_use = 30

        global_time_scale = 1.0
        if hasattr(batch, 'global_scale_factor'):
            global_time_scale = batch.global_scale_factor.mean().item()

        if not hasattr(batch, 'u') or batch.u is None:
            return True, []

        h_fixed_mins = float(batch.u[0, 2].item()) * global_time_scale
        num_orders = batch['order'].num_nodes
        num_ops = batch['operator'].num_nodes

        #prepare probs of each head
        p_act = out['activation'].view(-1).cpu().numpy()
        p_assign = out['assignment'].view(-1).cpu().numpy()
        p_seq = out['sequence'].view(-1).cpu().numpy()

        #adaptive thresholding (temperature scaling)
        #smooths out flat probabilities in large-scale graphs
        temperature = 1.0 + TEMPERATURE_SCALING_FACTOR * num_orders
        
        #safely inverse sigmoid to approximate logits, scale, and re-apply sigmoid
        p_assign_clipped = np.clip(p_assign, 1e-8, 1 - 1e-8)
        raw_logits = np.log(p_assign_clipped / (1 - p_assign_clipped))
        scaled_logits = raw_logits / temperature
        p_assign = 1 / (1 + np.exp(-scaled_logits))

        #edge indices (cpu for easier list processing)
        assign_idx = batch.edge_index_dict[('operator', 'assign', 'order')].cpu().numpy()
        seq_idx = batch.edge_index_dict[('order', 'to', 'order')].cpu().numpy()

        #processing time and base travel time (op -> order)
        proc_time_map = {}
        base_travel_map = {}
        assign_attr = batch.edge_attr_dict[('operator', 'assign', 'order')].cpu().numpy()
        for i in range(assign_idx.shape[1]):
            val_proc = float(assign_attr[i][0]) if assign_attr.ndim > 1 else float(assign_attr[i])
            val_travel = float(assign_attr[i][1]) if assign_attr.ndim > 1 and assign_attr.shape[1] > 1 else 0.0
            
            op = int(assign_idx[0, i])
            order = int(assign_idx[1, i])
            
            proc_time_map[(op, order)] = val_proc * global_time_scale
            base_travel_map[(op, order)] = val_travel * global_time_scale

        #travel time
        travel_time_map = {}
        seq_attr = batch.edge_attr_dict[('order', 'to', 'order')].cpu().numpy()
        for i in range(seq_idx.shape[1]):
            val = float(seq_attr[i][0]) if seq_attr.ndim > 1 else float(seq_attr[i])
            travel_time_map[(int(seq_idx[0, i]), int(seq_idx[1, i]))] = val * global_time_scale

        avg_travel_time = 1.0 
        if travel_time_map:
            avg_travel_time = sum(travel_time_map.values()) / len(travel_time_map)
            if avg_travel_time == 0: avg_travel_time = 1.0 

        #travel time (order -> order): (u, v) -> prob
        seq_adj = {}
        for i in range(seq_idx.shape[1]):
            u, v = int(seq_idx[0, i]), int(seq_idx[1, i])
            prob = float(p_seq[i])
            if u not in seq_adj: seq_adj[u] = []
            seq_adj[u].append((v, prob))

        for u in seq_adj:
            seq_adj[u].sort(key=lambda x: x[1], reverse=True) 

        all_mission_ids = batch['order'].global_id.cpu().tolist()
        all_operator_ids = batch['operator'].global_id.cpu().tolist()

        #activation
        active_ops = set(np.where(p_act >= self.act_threshold)[0])

        if use_extra_ops:
            inactive_ops = [op for op in range(num_ops) if op not in active_ops]
            best_inactive_ops = sorted(inactive_ops, key=lambda x: p_act[x], reverse=True)[:n_extra_ops_to_use]
            for best_inactive_op in best_inactive_ops:
                active_ops.add(best_inactive_op)

        if not active_ops:
            k = max(1, int(num_ops * 0.3))
            active_ops = set(np.argsort(p_act)[-k:].tolist())

        #STR- hungarian global assignment
        num_active = max(1, len(active_ops))
        #max_capacity_per_op = int(math.ceil((num_orders / num_active) * 1.5))
        max_capacity_per_op = int(math.ceil((num_orders / num_active) * 3.0))

        active_ops_list = list(active_ops)
        total_slots = len(active_ops_list) * max_capacity_per_op

        #initialize to 10.0 (high penalty because we are minimizing)
        cost_matrix = np.full((num_orders, total_slots), 10.0) 

        col_to_op = {}
        col_idx = 0
        for op in active_ops_list:
            for _ in range(max_capacity_per_op):
                col_to_op[col_idx] = op
                col_idx += 1

        #fill Cost Matrix
        for o in range(num_orders):
            for col in range(total_slots):
                op = col_to_op[col]
                mask = (assign_idx[0] == op) & (assign_idx[1] == o)
                idx_array = np.where(mask)[0]

                #if edge exists, use negative prob (so maximizing prob = minimizing cost)
                if len(idx_array) > 0:
                    prob = float(p_assign[idx_array[0]])
                    if prob > 0.0:
                        cost_matrix[o, col] = -prob 

        #print cost matrix
        #convert negative costs back to positive probabilities (and penalties to 0.0) for readability
        readable_matrix = np.where(cost_matrix == 10.0, 0.0, -cost_matrix)

        #create labels so you know exactly which row/column is which
        row_labels = [f"Order {o}" for o in range(num_orders)]
        col_labels = [f"Op {col_to_op[c]}" for c in range(total_slots)]

        #create a DataFrame and print it
        df_matrix = pd.DataFrame(readable_matrix, index=row_labels, columns=col_labels)

        print("\n" + "="*50)
        print("HUNGARIAN ASSIGNMENT PROBABILITY MATRIX")
        print("="*50)
        print(df_matrix.to_string(float_format=lambda x: f"{x:.3f}")) 
        print("="*50 + "\n")

        logging.info("="*50 + "\n")
        logging.info("HUNGARIAN ASSIGNMENT PROBABILITY MATRIX")
        logging.info(df_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
        logging.info("="*50 + "\n")

        #sequence matrix
        #create an n*n matrix initialized to 0.0
        seq_prob_matrix = np.zeros((num_orders, num_orders))

        #fill the matrix using the raw sequence probabilities
        for i in range(seq_idx.shape[1]):
            u, v = int(seq_idx[0, i]), int(seq_idx[1, i])
            prob = float(p_seq[i])
            seq_prob_matrix[u, v] = prob

        #create labels for the rows (from) and columns (to)
        order_labels = [f"Order {o}" for o in range(num_orders)]

        #create a DataFrame and print it
        df_seq_matrix = pd.DataFrame(seq_prob_matrix, index=order_labels, columns=order_labels)

        print("\n" + "="*70)
        print("SEQUENCE PROBABILITY MATRIX (Row = From, Col = To)")
        print("="*70)
        print(df_seq_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
        print("="*70 + "\n")

        logging.info("="*70 + "\n")
        logging.info("SEQUENCE PROBABILITY MATRIX (Row = From, Col = To)")
        logging.info(df_seq_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
        logging.info("="*70 + "\n")

        #solve global assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        current_clusters = {op: set() for op in range(num_ops)}
        ops_to_process = set()
        orders_to_assign = set(range(num_orders))

        for r, c in zip(row_ind, col_ind):
            #if cost < 0, a valid probability match was found
            if cost_matrix[r, c] < 0: 
                assigned_op = col_to_op[c]
                current_clusters[assigned_op].add(r)
                ops_to_process.add(assigned_op)
                orders_to_assign.remove(r)

        #route sequence builder
        final_op_routes = {op: [] for op in range(num_ops)} 
        rejected_orders = set(orders_to_assign) #include orders Hungarian algorithm couldn't place

        for op in ops_to_process:
            orders = list(current_clusters[op])
            if not orders: continue

            #sort cluster by assignment preference
            def get_prob(o):
                mask = (assign_idx[0] == op) & (assign_idx[1] == o)
                idx_array = np.where(mask)[0]
                return float(p_assign[idx_array[0]]) if len(idx_array) > 0 else 0.0

            orders.sort(key=lambda o: get_prob(o), reverse=True)

            unvisited = set(orders)
            route_steps = []
            current_time = 0.0
            curr = None

            #initial start
            for cand in orders:
                t_travel = base_travel_map.get((op, cand), avg_travel_time) 
                t_proc = proc_time_map.get((op, cand), 0.0)

                if (t_proc + t_travel) <= h_fixed_mins:
                    curr = cand
                    route_steps.append({
                        "mission_id": all_mission_ids[curr],
                        "_internal": curr,
                        "start_time": round(t_travel, 2),
                        "finish_time": round(t_travel + t_proc, 2),
                        "processing_duration": round(t_proc, 2),
                        "travel_duration": round(t_travel, 2),
                        "successor": None
                    })
                    current_time = t_proc + t_travel
                    unvisited.discard(curr)
                    break

            #route using sequence probs
            if curr is not None:
                while unvisited:
                    best_next = None
                    best_metrics = None

                    if curr in seq_adj:
                        for neighbor, _ in seq_adj[curr]:
                            if neighbor in unvisited:
                                t_travel = travel_time_map.get((curr, neighbor), 0.0)
                                t_proc = proc_time_map.get((op, neighbor), 0.0)
                                finish = current_time + t_travel + t_proc

                                if finish <= h_fixed_mins:
                                    best_next = neighbor
                                    best_metrics = (t_travel, t_proc, finish)
                                    break

                    if not best_next:
                        for neighbor in unvisited:
                            t_travel = travel_time_map.get((curr, neighbor), avg_travel_time)
                            if t_travel == 0.0: t_travel = avg_travel_time 
                            t_proc = proc_time_map.get((op, neighbor), 0.0)
                            finish = current_time + t_travel + t_proc

                            if finish <= h_fixed_mins:
                                best_next = neighbor
                                best_metrics = (t_travel, t_proc, finish)
                                break

                    if best_next:
                        t, p, f = best_metrics
                        route_steps[-1]["successor"] = all_mission_ids[best_next]
                        route_steps.append({
                            "mission_id": all_mission_ids[best_next],
                            "_internal": best_next,
                            "start_time": round(current_time + t, 2),
                            "finish_time": round(f, 2),
                            "processing_duration": round(p, 2),
                            "travel_duration": round(t, 2),
                            "successor": None
                        })
                        current_time = f
                        curr = best_next
                        unvisited.discard(curr)
                    else:
                        break

            final_op_routes[op] = route_steps

            assigned_in_route = {s["_internal"] for s in route_steps}
            for o in orders:
                if o not in assigned_in_route:
                    rejected_orders.add(o)

        #STR- tail insertion for rejected/unassigned orders
        if rejected_orders:
            print(f"Tail-insertion pass: {len(rejected_orders)} orders still unassigned")
            logging.info(f"Tail-insertion pass: {len(rejected_orders)} orders still unassigned")

            predicted_schedule_name = filename.replace(".json", "")
            if not predicted_schedule_name in self.batch_tail_insertions.keys():
                self.batch_tail_insertions[predicted_schedule_name] = len(rejected_orders)

            for o in list(rejected_orders):
                best_op = None
                best_finish = None
                best_t_travel = 0.0 
                best_t_proc = 0.0 

                for op in active_ops: 
                    route = final_op_routes.get(op, [])
                    curr_finish = route[-1]["finish_time"] if route else 0.0
                    last_node = route[-1]['_internal'] if route else None

                    #correctly pull travel time from base (-1) if route is empty
                    if last_node is not None:
                        t_travel = travel_time_map.get((last_node, o), avg_travel_time)
                    else:
                        t_travel = base_travel_map.get((op, o), avg_travel_time)

                    t_proc = proc_time_map.get((op, o), 0.0)
                    finish = curr_finish + t_travel + t_proc

                    if finish <= h_fixed_mins:
                        if best_finish is None or finish < best_finish:
                            best_finish = finish
                            best_op = op
                            best_t_travel = t_travel 
                            best_t_proc = t_proc 

                if best_op is not None:
                    route = final_op_routes[best_op]
                    #the start of the travel is the finish time of the previous order (or 0.0 if first)
                    travel_start_time = route[-1]["finish_time"] if route else 0.0
                    step = {
                        "mission_id": all_mission_ids[o],
                        "_internal": o,
                        "start_time": round(travel_start_time + best_t_travel, 2), 
                        "finish_time": round(travel_start_time + best_t_travel + best_t_proc, 2), 
                        "processing_duration": round(best_t_proc, 2),
                        "travel_duration": round(best_t_travel, 2), 
                        "successor": None,
                    }
                    if route:
                        route[-1]["successor"] = all_mission_ids[o]
                    route.append(step)
                    rejected_orders.remove(o)
                    print(f"Tail-inserted order {o} to Op {best_op}")
                    logging.info(f"Tail-inserted order {o} to Op {best_op}")

        # --- 5. SAFE sequence refinement (Makespan Minimization) ---
        for op, route in final_op_routes.items():
            if len(route) <= 2:
                continue 

            cluster_nodes = [s['_internal'] for s in route]
            current_best_makespan = route[-1]['finish_time']
            best_route_steps = list(route) 

            def get_travel(u, v):
                if u == -1:
                    return base_travel_map.get((op, v), avg_travel_time)
                t = travel_time_map.get((u, v), avg_travel_time)
                if t == 0.0: 
                    t = avg_travel_time
                return t

            def evaluate_makespan(rt_nodes):
                current_time = 0.0
                for i, cand in enumerate(rt_nodes):
                    prev = rt_nodes[i-1] if i > 0 else -1
                    t_travel = get_travel(prev, cand)
                    t_proc = proc_time_map.get((op, cand), 0.0)
                    finish = current_time + t_travel + t_proc

                    if finish > h_fixed_mins:
                        return False, float('inf')
                    current_time = finish
                return True, current_time

            def build_steps_for_nodes(rt_nodes):
                current_time = 0.0
                steps = []
                for i, cand in enumerate(rt_nodes):
                    prev = rt_nodes[i-1] if i > 0 else -1
                    t_travel = get_travel(prev, cand)
                    t_proc = proc_time_map.get((op, cand), 0.0)
                    finish = current_time + t_travel + t_proc

                    steps.append({
                        'mission_id': all_mission_ids[cand],
                        '_internal': cand,
                        'start_time': round(current_time + t_travel, 2),
                        'finish_time': round(finish, 2),
                        'processing_duration': round(t_proc, 2),
                        'travel_duration': round(t_travel, 2),
                        'successor': None
                    })
                    current_time = finish

                for i in range(len(steps) - 1):
                    steps[i]['successor'] = steps[i+1]['mission_id']
                return steps

            improved = True
            while improved:
                improved = False
                for i in range(len(cluster_nodes)):
                    for j in range(i + 1, len(cluster_nodes)):

                        # 1. Try Point Swap 
                        new_nodes_swap = cluster_nodes[:]
                        new_nodes_swap[i], new_nodes_swap[j] = new_nodes_swap[j], new_nodes_swap[i]

                        is_feas_swap, new_makespan_swap = evaluate_makespan(new_nodes_swap)
                        if is_feas_swap and new_makespan_swap < current_best_makespan - 1e-3:
                            current_best_makespan = new_makespan_swap
                            cluster_nodes = new_nodes_swap
                            best_route_steps = build_steps_for_nodes(cluster_nodes)
                            improved = True
                            continue

                        # 2. Try 2-Opt Segment Reversal
                        new_nodes_rev = cluster_nodes[:i] + cluster_nodes[i:j+1][::-1] + cluster_nodes[j+1:]

                        is_feas_rev, new_makespan_rev = evaluate_makespan(new_nodes_rev)
                        if is_feas_rev and new_makespan_rev < current_best_makespan - 1e-3:
                            current_best_makespan = new_makespan_rev
                            cluster_nodes = new_nodes_rev
                            best_route_steps = build_steps_for_nodes(cluster_nodes)
                            improved = True
                            continue

            final_op_routes[op] = best_route_steps

        #reconstruct output schedule
        schedule_data = {
            "metadata": {
                "num_orders": int(num_orders),
                "num_operators": int(num_ops),
                "valid": True,
                "schedule_id": getattr(batch, 'schedule_id', ['unknown'])[0] if hasattr(batch, 'schedule_id') else 'unknown',
            },
            "operators": []
        }

        assigned_count = 0
        for op_idx, route in final_op_routes.items():
            if route:
                clean_route = [{k:v for k,v in s.items() if k!="_internal"} for s in route]
                schedule_data["operators"].append({
                    "operator_id": all_operator_ids[op_idx],
                    "assigned_orders_count": len(clean_route),
                    "routes": [clean_route]
                })
                assigned_count += len(clean_route)

        activate_extra_op = False
        if assigned_count < num_orders and len(active_ops) < num_ops:
            print(f"Warning: {num_orders - assigned_count} orders unassigned.")
            logging.critical(f"Warning: {num_orders - assigned_count} orders unassigned.")

            activate_extra_op = True

            predicted_schedule_name = filename.replace(".json", "")
            if not predicted_schedule_name in self.batch_tail_insertions_with_new_activations.keys():
                self.batch_tail_insertions_with_new_activations[predicted_schedule_name] = num_orders - assigned_count

        if activate_extra_op and len(active_ops) + n_extra_ops_to_use < np.size(p_act):
            n_extra_ops_to_use = n_extra_ops_to_use + 1
            print(f"Trying to resolve by activating extra {n_extra_ops_to_use} operators.")
            logging.info(f"Trying to resolve by activating extra {n_extra_ops_to_use} operators.")

            self.export_schedule_with_timings_v3_hungarian_seq_refined(batch=batch, out=out, filename=filename, use_extra_ops=True, n_extra_ops_to_use=n_extra_ops_to_use)
        else:
            unassigned_orders = num_orders - assigned_count
            schedule_data["metadata"]["unassigned_orders"] = unassigned_orders
            if unassigned_orders > 0:
                print(colored_background_str(r=255, g=0, b=5, text=f"Warning: {unassigned_orders} orders remain unassigned."))
                logging.CRITICAL(f"Warning: {unassigned_orders} orders remain unassigned.")

            h_valid, h_violations = self.check_horizon_constraint(batch, schedule_data, global_time_scale)

            schedule_data["metadata"]["horizon_valid"] = h_valid
            schedule_data["metadata"]["horizon_violations"] = h_violations

            os.makedirs(os.path.dirname(self.predicted_schedule_dir), exist_ok=True)
            with open(os.path.join(self.predicted_schedule_dir, filename.replace('Batch', 'schedule')), 'w') as f:
                json.dump(schedule_data, f, indent=4)

            print(f"Schedule exported with name: {filename}")
            logging.info(f"Schedule exported with name: {filename}")
    
    @torch.no_grad()
    def export_autoregressive_schedule(self, model, batch, filename="ar_schedule.json"):
        """
        True autoregressive decoding for inference.
        Iteratively runs the GNN, picks the best valid operator-order assignment,
        updates the dynamic state, and repeats until all orders are assigned.
        Directly exports the JSON schedule.
        """
        model.eval()
        device = batch.edge_index_dict[('operator', 'assign', 'order')].device

        global_time_scale = 1.0
        if hasattr(batch, 'global_scale_factor'):
            global_time_scale = batch.global_scale_factor.mean().item()

        num_ops = batch['operator'].x.size(0)
        num_orders = batch['order'].x.size(0)
        all_mission_ids = batch['order'].global_id.cpu().tolist()
        all_operator_ids = batch['operator'].global_id.cpu().tolist()
        
        #initialize Dynamic Features (Step 0)
        op_dynamic = torch.zeros((num_ops, 6), device=device)
        order_dynamic = torch.zeros((num_orders, 1), device=device)
        
        op_batch_idx = batch['operator'].batch if hasattr(batch['operator'], 'batch') else torch.zeros(num_ops, dtype=torch.long, device=device)
        order_batch_idx = batch['order'].batch if hasattr(batch['order'], 'batch') else torch.zeros(num_orders, dtype=torch.long, device=device)
        
        #init remaining_h_fixed (multiply by scale to match feature space)
        op_dynamic[:, 0] = batch.u[op_batch_idx, 2]
        
        #original Time Constraints in Minutes
        h_fixed_mins = float(batch.u[0, 2].item()) * global_time_scale
        
        assign_src = batch['operator', 'assign', 'order'].edge_index[0]
        assign_dst = batch['operator', 'assign', 'order'].edge_index[1]
        assign_attrs = batch['operator', 'assign', 'order'].edge_attr

        seq_src = batch['order', 'to', 'order'].edge_index[0].cpu().numpy()
        seq_dst = batch['order', 'to', 'order'].edge_index[1].cpu().numpy()
        seq_attr = batch['order', 'to', 'order'].edge_attr.cpu().numpy()
        
        #pre-build travel time map for orders
        travel_time_map = {}
        for i in range(len(seq_src)):
            val = float(seq_attr[i][0]) if seq_attr.ndim > 1 else float(seq_attr[i])
            travel_time_map[(int(seq_src[i]), int(seq_dst[i]))] = val * global_time_scale

        avg_travel_time = sum(travel_time_map.values()) / max(1, len(travel_time_map))
        if avg_travel_time == 0: avg_travel_time = 1.0 
        
        base_travel_map = {}
        for i in range(len(assign_src)):
            op = int(assign_src[i])
            order = int(assign_dst[i])
            val_travel = float(assign_attrs[i][1]) if assign_attrs.size(1) > 1 else 0.0
            base_travel_map[(op, order)] = val_travel * global_time_scale

        unassigned_orders = set(range(num_orders))
        
        #state tracking for json building
        final_op_routes = {op: [] for op in range(num_ops)}
        current_time_per_op = {op: 0.0 for op in range(num_ops)}

        step = 0
        max_steps = num_orders + 10 
        
        print(f"Starting AR Decoding for {num_orders} orders...")

        while unassigned_orders and step < max_steps:
            step += 1
            
            for i in range(batch.u.size(0)):
                mask = (op_batch_idx == i)
                if mask.sum() > 0:
                    max_workload = op_dynamic[mask, 1].max()
                    op_dynamic[mask, 2] = max_workload - op_dynamic[mask, 1]

            x_dict_dynamic = {
                'operator': torch.cat([batch['operator'].x, op_dynamic], dim=1),
                'order': torch.cat([batch['order'].x, order_dynamic], dim=1)
            }
            batch_dict_arg = {'operator': op_batch_idx, 'order': order_batch_idx}

            preds = model(
                x_dict_dynamic, 
                batch.edge_index_dict, 
                batch.edge_attr_dict,
                batch.u,
                batch_dict=batch_dict_arg
            )

            p_assign = preds['action'].view(-1)
            
            #mask out invalid moves
            valid_edge_mask = torch.ones_like(p_assign, dtype=torch.bool)
            for e_idx in range(len(assign_dst)):
                chosen_op = assign_src[e_idx].item()
                chosen_order = assign_dst[e_idx].item()
                
                if chosen_order not in unassigned_orders:
                    valid_edge_mask[e_idx] = False
                    continue
                    
                proc_time_feat = assign_attrs[e_idx, 0].item()
                
                #calculate real travel time (from base if first order, or from previous order)
                if len(final_op_routes[chosen_op]) == 0:
                    real_travel_time = base_travel_map.get((chosen_op, chosen_order), avg_travel_time)
                else:
                    prev_order = final_op_routes[chosen_op][-1]['_internal']
                    real_travel_time = travel_time_map.get((prev_order, chosen_order), avg_travel_time)
                    
                real_proc_time = proc_time_feat * global_time_scale
                
                #check if it fits in real time constraint
                if current_time_per_op[chosen_op] + real_travel_time + real_proc_time > h_fixed_mins:
                    valid_edge_mask[e_idx] = False

            if not valid_edge_mask.any():
                print(f"Decoding halted early at step {step}: Operators have no remaining time.")
                break

            p_assign_masked = p_assign.clone()
            p_assign_masked[~valid_edge_mask] = -1.0

            best_edge_idx = torch.argmax(p_assign_masked).item()
            best_prob = p_assign_masked[best_edge_idx].item()
            
            if best_prob < 0: 
                break

            chosen_op = assign_src[best_edge_idx].item()
            chosen_order = assign_dst[best_edge_idx].item()
            proc_time_feat = assign_attrs[best_edge_idx, 0].item()
            travel_time_feat = assign_attrs[best_edge_idx, 1].item() if assign_attrs.size(1) > 1 else 0.0

            #calculate Actual Schedule Timings
            if len(final_op_routes[chosen_op]) == 0:
                t_travel = base_travel_map.get((chosen_op, chosen_order), avg_travel_time)
            else:
                prev_order = final_op_routes[chosen_op][-1]['_internal']
                t_travel = travel_time_map.get((prev_order, chosen_order), avg_travel_time)
                
            t_proc = proc_time_feat * global_time_scale
            
            start_time = current_time_per_op[chosen_op] + t_travel
            finish_time = start_time + t_proc
            
            #record Step in Route
            step_dict = {
                "mission_id": all_mission_ids[chosen_order],
                "_internal": chosen_order,
                "start_time": round(start_time, 2),
                "finish_time": round(finish_time, 2),
                "processing_duration": round(t_proc, 2),
                "travel_duration": round(t_travel, 2),
                "successor": None
            }
            
            if len(final_op_routes[chosen_op]) > 0:
                final_op_routes[chosen_op][-1]["successor"] = all_mission_ids[chosen_order]
                
            final_op_routes[chosen_op].append(step_dict)
            current_time_per_op[chosen_op] = finish_time
            unassigned_orders.remove(chosen_order)

            #update GNN dynamic features
            order_dynamic[chosen_order, 0] = 1.0 
            op_dynamic[chosen_op, 0] -= (proc_time_feat + travel_time_feat) 
            op_dynamic[chosen_op, 1] += (proc_time_feat + travel_time_feat) 
            op_dynamic[chosen_op, 3] = batch['order'].x[chosen_order, 4] 
            op_dynamic[chosen_op, 4] = batch['order'].x[chosen_order, 5] 
            op_dynamic[chosen_op, 5] = 1.0 

        #bBuild json output structure
        os.makedirs(self.predicted_schedule_dir, exist_ok=True)
        final_filepath = os.path.join(self.predicted_schedule_dir, filename)

        schedule_id = filename.replace('.json', '')
        
        schedule_json = {
            "metadata": {
                "num_orders": num_orders,
                "num_operators": num_ops,
                "valid": len(unassigned_orders) == 0,
                "schedule_id": schedule_id,
                "unassigned_orders": len(unassigned_orders),
                "horizon_valid": True,  #always guaranteed by the while loop logic
                "horizon_violations": []
            },
            "operators": []
        }
        
        for op_id, route in final_op_routes.items():
            if not route:
                continue
                
            operator_data = {
                "operator_id": all_operator_ids[op_id],
                "assigned_orders_count": len(route),
                "workload_mins": round(route[-1]["finish_time"], 2) if route else 0.0,
                "routes": [[]] 
            }
            
            for step_data in route:
                clean_step = step_data.copy()
                del clean_step["_internal"] 
                operator_data["routes"][0].append(clean_step)
                
            schedule_json["operators"].append(operator_data)
        
        with open(final_filepath, 'w') as f:
            json.dump(schedule_json, f, indent=4)

        print(f"Schedule exported with name: {filename}")

def decode_non_autoregressive(model, loader, scheduleDecoder, device='cuda'):
    idx = 0
    for batch in loader:
        batch = batch.to(device)
        batch_dict = {'operator': batch['operator'].batch, 'order': batch['order'].batch}

        #op_embd = schedule_evaluator.diagnose_operator_embeddings(model, batch)

        print(f"Evaluating schedule_id: {batch.schedule_id[0]}")
        logging.info(f"Evaluating schedule_id: {batch.schedule_id[0]}")

        if batch['order'].num_nodes < 2:
            print(f"Schedule [{idx}] has less than 2 orders, skipping feasibility/export check.")
            logging.info(f"Schedule [{idx}] has less than 2 orders, skipping feasibility/export check.")
            continue

        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.u, batch_dict=batch_dict)

        is_valid, report = scheduleDecoder.evaluate_full_feasibility(batch, out)

        idx = idx + 1
        #scheduleValidator.export_schedule_with_timings_v2(batch, report, out, filename=f"predicted_{batch.schedule_id[0]}.json")
        #scheduleDecoder.export_schedule_with_timings_v3(batch, out, filename=f"predicted_{batch.schedule_id[0]}.json")
        #scheduleDecoder.export_schedule_with_timings_v3_hungarian(batch, out, filename=f"predicted_{batch.schedule_id[0]}.json")
        scheduleDecoder.export_schedule_with_timings_v3_hungarian_seq_refined(batch, out, filename=f"predicted_{batch.schedule_id[0]}.json")

        #print(report)
        if is_valid:
            print(f"Schedule [{idx}] is Feasible!")
            logging.info(f"Schedule [{idx}] is Feasible!")
            #safe to calculate optimality gap
            pass
        else:
            print(colored_background_str(r=255, g=0, b=5, text=f"Schedule [{idx}] is NOT Feasible!"))
            logging.CRITICAL(f"Schedule [{idx}] is NOT Feasible!")
            print(f"Activation Feasibility: {report['act_ok']} with stats {report['act_errs']}")

            if not report["seq_ok"]:
                print(f"Sequence Invalid: {report['seq_errs']}")

def decode_autoregressive(model, loader, scheduleDecoder, device='cuda'):
    """
    True autoregressive decoding for inference.
    Iteratively runs the GNN, picks the best valid operator-order assignment,
    updates the dynamic state, and repeats until all orders are assigned 
    or all operators terminate.
    Estimates masks matching the expected format of the downstream export_schedule pipeline.
    """

    for batch_idx, batch in enumerate(loader):
        print(f"Evaluating schedule_id: {batch.schedule_id[0]}")
        batch = batch.to(device)
        scheduleDecoder.export_autoregressive_schedule(model, batch, filename=f"predicted_{batch.schedule_id[0]}.json")

if __name__ == "__main__":
    use_large_scale = False
    use_autoregressive_model = False

    if use_large_scale:
        #init large-scale dataset
        dataset = GnnScheduleDataset(
            schedule_dir=None,
            mission_base_path=MISSION_LARGE_BATCH_DIR,
            edge_base_path=MISSION_LARGE_BATCH_TRAVEL_DIR,
            pallet_types_file_path=UDC_TYPES_DIR,
            fork_path=FORK_LIFTS_DIR.replace('10W', '100W'),
            large_batch_dir=LARGE_BATCH_DIR,
            large_batch_travel_dir=LARGE_BATCH_TRAVEL_DIR
        )
    else:
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
        'hidden_dim': 128,
        'heads': 4,
        'dropout': 0.0,
        'lr_trunk': 0.001,
        'lr_activation': 0.01,
        'lr_assignment': 0.0005,
        'lr_sequence': 0.001
    }
    
    # #tuned thresholds for feasibility validation (B100)
    # best_thresholds =  {
    #     'activation': 0.2137,
    #     'assignment': 0.0321,
    #     'sequence': 0.1096
    # }

    #tuned thresholds for feasibility validation (B1000)
    # best_thresholds =  {
    #     'activation': 0.7252,
    #     'assignment': 0.1126,
    #     'sequence': 0.1093
    # }

    #coupled tasks - pre-tuned thresholds for feasibility validation (B1000)
    # best_thresholds =  {
    #     'activation': 0.6251,
    #     'assignment': 0.1000,
    #     'sequence': 0.1153
    # }

    #coupled tasks - tuned thresholds for feasibility validation (B1000) droupout 0.2
    # best_thresholds =  {
    #     'activation': 0.1697,
    #     'assignment': 0.0469,
    #     'sequence': 0.1163
    # }

    #coupled tasks - tuned thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.7248,
    #     'assignment': 0.0856,
    #     'sequence': 0.1046
    # }

    # #coupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.4472,
    #     'assignment': 0.0678,
    #     'sequence': 0.1228
    # }

    # #coupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1 - mean aggregation
    # best_thresholds =  {
    #     'activation': 0.4992,
    #     'assignment': 0.0785,
    #     'sequence': 0.1168
    # }

    # #coupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1 - mean aggregation - coupling dropout
    # best_thresholds =  {
    #     'activation': 0.4457,
    #     'assignment': 0.0731,
    #     'sequence': 0.0848
    # }

    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.4777,
    #     'assignment': 0.0847,
    #     'sequence': 0.1166
    # }

    #with order embedding
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.5740,
    #     'assignment': 0.0865,
    #     'sequence': 0.1043
    # }

    #with monotic id
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.5250,
    #     'assignment': 0.0799,
    #     'sequence': 0.0935
    # }

    #with monotic id & capacity penalty
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.5457,
    #     'assignment': 0.3216,
    #     'sequence': 0.1089
    # }

    #with monotic id & capacity penalty & affinity score
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.6688,
    #     'assignment': 0.3036,
    #     'sequence': 0.0981
    # }

    #capacity penalty & affinity score + No monotic id & No PE
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.3549,
    #     'assignment': 0.0789,
    #     'sequence': 0.1094
    # }

    #with normalized monotic id & capacity penalty & affinity score
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.4816,
    #     'assignment': 0.0730,
    #     'sequence': 0.1057
    # }

    #heuristic-boost capacity penalty + no alpha/beta loss
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.5098,
    #     'assignment': 0.0789,
    #     'sequence': 0.0970
    # }

    #heuristic-boost capacity penalty + no alpha/beta loss + no assignment monotonic id
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.5284,
    #     'assignment': 0.0813,
    #     'sequence': 0.0926
    # }


    #Augumented data - heuristic-boost capacity penalty + no alpha/beta loss + no assignment monotonic id
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    # best_thresholds =  {
    #     'activation': 0.5147,
    #     'assignment': 0.0829,
    #     'sequence': 0.0836
    # }

    #Augumented data (4 augmentations) - heuristic-boost capacity penalty + no alpha/beta loss + no assignment monotonic id
    #fallback to decoupled tasks - tuned heuristic-boost thresholds for feasibility validation (B1000) droupout 0.1
    best_thresholds =  {
        'activation': 0.4813,
        'assignment': 0.0907,
        'sequence': 0.0858
    }

    if use_autoregressive_model:
        model = MultiCriteriaGNNModel_AutoRegressive(
            hidden_dim=best_conf.get('hidden_dim', 64),
            heads=best_conf.get('heads', 4),
        ).to(device)
    else:
        model = MultiCriteriaGNNModel(
            metadata=sample_data.metadata(),
            hidden_dim=best_conf['hidden_dim'],
            num_layers=3,
            heads=best_conf['heads'],
            dropout=best_conf['dropout']
        ).to(device)
    
    model.load_model(device=device) #load pre-trained model weights

    totals = {"n": 0, "act_f1": 0.0, "assign_f1": 0.0, "seq_f1": 0.0,
              "feasible": 0, "gap_sum": 0.0}
    
    BATCH_SIZE = 1 #we want to evaluate one instance for coherency with mip solver schedules
    schedule_evaluator = ScheduleEvaluator(model=model,
                                        schedule_dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        act_threshold=best_thresholds['activation'],
                                        assign_threshold=best_thresholds['assignment'],
                                        seq_threshold=best_thresholds['sequence'],
                                        split_train_validation=not use_large_scale,
                                        use_spatial_augmentation=use_large_scale
                                        )
    
    loader = DataLoader(schedule_evaluator.schedule_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    scheduleDecoder = ScheduleDecoder(
        act_threshold=best_thresholds['activation'],
        assign_threshold=best_thresholds['assignment'],
        seq_threshold=best_thresholds['sequence'],
        predicted_schedule_dir=PREDICTED_LARGE_SCHEDULE_DIR if use_large_scale else PREDICTED_SCHEDULE_DIR
    )
    
    print("Starting Schedule Validation - total batches:", len(loader))

    if use_autoregressive_model:
        decode_autoregressive(model, loader, scheduleDecoder, device)
    else:
        decode_non_autoregressive(model, loader, scheduleDecoder, device)

    tail_insertions = str(scheduleDecoder.batch_tail_insertions).replace(",", ",\n")
    
    print(f"Batch Tail Insertions: {tail_insertions}")
    logging.info(f"Batch Tail Insertions: {tail_insertions}")

    additional_act_tail_insertions = str(scheduleDecoder.batch_tail_insertions_with_new_activations).replace(",", ",\n")
    print(f"Batch Additional Activation Tail Insertions: {additional_act_tail_insertions}")
    logging.info(f"Batch Additional Activation Tail Insertions: {additional_act_tail_insertions}")

    overall_tail_insertions = dict()
    overall_tail_insertions["tail_insertions"] = scheduleDecoder.batch_tail_insertions
    overall_tail_insertions["additional_act_tail_insertions"] = scheduleDecoder.batch_tail_insertions_with_new_activations

    #save tail insertions
    target_tail_insertions_dir = LARGE_BATCH_TAIL_INSERTIONS_DIR if use_large_scale else MINI_BATCH_TAIL_INSERTIONS_DIR
    with open(target_tail_insertions_dir, 'w') as f:
        json.dump(overall_tail_insertions, f, indent=4)
        print(f"Saved tail insertions to {target_tail_insertions_dir}")
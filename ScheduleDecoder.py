import networkx as nx
import torch
import numpy as np
import os
import itertools
import json
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from ScheduleEvaluator import ScheduleEvaluator
from MultiCriteriaGNNModel import MultiCriteriaGNNModel
from GnnScheduleDataset import GnnScheduleDataset

LARGE_SCALE_BATCH_NAME = "Batch10000M" #Batch1000M, Batch9000M or Batch10000M
TARGET_MINI_BATCH_SIZE = 10 #number of missions per mini-batch
LARGE_SCALE_MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}.csv"
PREPROCESSED_BATCH_DIR = f"./preprocessed/{LARGE_SCALE_BATCH_NAME}/Batch{TARGET_MINI_BATCH_SIZE}M_idx.xlsx" #idx to be replaced cluster idx
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts10W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = f"./schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
PREDICTED_SCHEDULE_DIR = f"./predicted_schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
BATCH_SIZE = 32 #nice to be equal to 32 or 64 since we have small mini-batch instances
H_FIXED_EXCEED_TOLERANCE_MIN = 0.0 #allow schedules to tolerate H_fixed exceedance 
MAX_ITERATIONS_PER_ORDER = 10 #max attempts to find a feasible operator for an order based on assignment probs (in iterative repair)

#default threshold for binary classification accurcy like logistic regression after sigmoid
#need to be tuned if the classes are imbalanced (can be relevated from classification report / roc curve)
CLASSIFICATION_THRESHOLD = 0.05

class ScheduleDecoder:
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
        print(f"Activation Probabilities (sample): {out['activation'].view(-1)[:10].cpu().detach().numpy()}")

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
        os.makedirs(os.path.dirname(PREDICTED_SCHEDULE_DIR), exist_ok=True)

        #save schedule
        with open(os.path.join(PREDICTED_SCHEDULE_DIR, filename), 'w') as f:
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
        os.makedirs(os.path.dirname(PREDICTED_SCHEDULE_DIR), exist_ok=True)
        
        #save schedule
        with open(os.path.join(PREDICTED_SCHEDULE_DIR, filename), 'w') as f:
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

        #travel time (Order -> Order): (u, v) -> t time
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
        os.makedirs(os.path.dirname(PREDICTED_SCHEDULE_DIR), exist_ok=True)
        
        #save schedule
        with open(os.path.join(PREDICTED_SCHEDULE_DIR, filename), 'w') as f:
            json.dump(schedule_data, f, indent=4)
        
        print(f"Schedule exported with name: {filename}")


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
    best_thresholds =  {
        'activation': 0.7252,
        'assignment': 0.1126,
        'sequence': 0.1093
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

    scheduleValidator = ScheduleDecoder(
        act_threshold=best_thresholds['activation'],
        assign_threshold=best_thresholds['assignment'],
        seq_threshold=best_thresholds['sequence']
    )
    
    print("Starting Schedule Validation - total batches:", len(loader))

    idx = 0
    for batch in loader:
        batch = batch.to(device)
        batch_dict = {'operator': batch['operator'].batch, 'order': batch['order'].batch}

        #op_embd = schedule_evaluator.diagnose_operator_embeddings(model, batch)

        print(f"Evaluating schedule_id: {batch.schedule_id[0]}")

        if batch['order'].num_nodes < 2:
            print(f"Schedule [{idx}] has less than 2 orders, skipping feasibility/export check.")
            continue

        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.u, batch_dict=batch_dict)

        is_valid, report = scheduleValidator.evaluate_full_feasibility(batch, out)

        idx = idx + 1
        scheduleValidator.export_schedule_with_timings_v2(batch, report, out, filename=f"predicted_{batch.schedule_id[0]}.json")
        
        #print(report)
        if is_valid:
            print(f"Schedule [{idx}] is Feasible!")
            #safe to calculate optimality gap
            pass
        else:
            print(f"Schedule [{idx}] is NOT Feasible!")
            print(f"Activation Feasibility: {report['act_ok']} with stats {report['act_errs']}")

            if not report["seq_ok"]:
                print(f"Sequence Invalid: {report['seq_errs']}")
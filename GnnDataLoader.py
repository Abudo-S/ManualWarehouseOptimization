import pandas as pd
import torch
#from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import os
import re
from ParameterDataLoader import ParameterDataLoader
from MultiCriteriaGNNModel import MultiCriteriaGNNModel

#file paths
MISSION_BATCH_DIR = "./datasets/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = "./datasets/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts10W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = "./schedules/mini-batch/"

H_FIXED_MINUTES = 60 #480 for base shift 
ALPHA = 1.0 #makespan weight
BETA = 1000.0 #operator activation weight (ex. 1000 = fully oriented to operator activation, 50 = balanced)
BIG_M = 1e5

class GnnDataLoader:
    
    def parse_filename_params(self, filename):
        """
        Extracts Global Parameters A (alpha), B (beta), H (H_fixed) from  schedule_file filename.
        pattern: schedule10M_1_A1.0_B100.0_H90.csv
        """
        pattern = r"A(?P<A>[\d.]+)_B(?P<B>[\d.]+)_H(?P<H>\d+)"
        match = re.search(pattern, filename)

        assert match, f"Can't extract global params from '{filename}'"

        alpha = float(match.group('A'))
        beta = float(match.group('B'))
        h_fixed = float(match.group('H'))
        
        return alpha, beta, h_fixed

    def load_and_process_data(
        self,
        node_file_path,
        pallet_types_file_path, 
        operator_file_path, 
        edge_file_path, 
        schedule_file_path
    ):
        #parse global parameters
        filename = os.path.basename(schedule_file_path)
        alpha, beta, h_fixed = self.parse_filename_params(filename)
        print(f"Global params extracted from[{filename}]: Alpha={alpha}, Beta={beta}, H_fixed={h_fixed}")

        #load dataframe
        df_missions = pd.read_csv(node_file_path)
        df_pallet_types = pd.read_csv(pallet_types_file_path)
        df_ops = pd.read_csv(operator_file_path)
        df_edges = pd.read_csv(edge_file_path)
        df_schedule = pd.read_csv(schedule_file_path)

        #concatenate pallet type features in mission features
        df_missions = df_missions.merge(df_pallet_types[['TP_UDC', 'WIDTH', 'LENGTH']], on='TP_UDC', how='left')
        
        #node idx mappings
        mission_ids = df_missions['CD_MISSION'].unique()
        order_map = {id: i for i, id in enumerate(mission_ids)}
        num_mission = len(mission_ids)
        
        op_ids = df_ops['OID'].unique()
        op_map = {id: i for i, id in enumerate(op_ids)}
        num_ops = len(op_ids)
        
        #actual_assignments = set(zip(df_schedule['Operator_ID'], df_schedule['To_Node']))

        scaler = MinMaxScaler()
        
        #-STR-node feature engineering

        #mission features [pallet dims, dest axes]
        missions_features = [
            'WEIGHT', 'HEIGHT', 'WIDTH', 'LENGTH', 'FROM_X', 'FROM_Y', 'FROM_Z','TO_X', 'TO_Y', 'TO_Z'
        ]
        mission_feats_raw = df_missions[missions_features].fillna(0)
        missions_scaled = scaler.fit_transform(mission_feats_raw)
        x_missions = torch.tensor(missions_scaled, dtype=torch.float)

        #operator features [Speed, Fork dims]
        operator_features = [
            'SPEED', 'UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD', 'FORK_WIDTH', 'FORK_LENGTH'
        ]
        op_feats_raw = df_ops[operator_features].fillna(0).values
        
        #add H_fixed column (same for all ops in this batch, based on filename)
        #if H is dynamic per operator, we'd read it from csv. Here it's global per file.
        # op_feats_norm = scaler.fit_transform(op_feats_raw)
        # h_fixed_col = np.full((num_ops, 1), h_fixed)
        
        # x_ops = torch.cat([
        #     torch.tensor(op_feats_norm, dtype=torch.float),
        #     torch.tensor(h_fixed_col, dtype=torch.float)
        # ], dim=1)
        
        #scale Z_features since they are not real in our dataset
        features_to_scale = ['FROM_Z','TO_Z']
        df_to_scale = df_missions[features_to_scale]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_to_scale)

        df_scaled_features = pd.DataFrame(
            scaled_data,
            columns=features_to_scale,
            index=df_missions.index
        )

        df_scaled_features = df_scaled_features.clip(lower=0)
        #df_scaled_features.head()

        df_unscaled_features = df_missions.drop(columns=features_to_scale)

        mission_batch_df_scaled = pd.concat([df_unscaled_features, df_scaled_features], axis=1)

        ops_scaled = scaler.fit_transform(op_feats_raw)
        x_ops = torch.tensor(ops_scaled, dtype=torch.float)

        BASE_MISSION = [0 for _ in range(len(mission_batch_df_scaled.columns))]
        df_missions_batch_with_base = pd.concat([pd.DataFrame([BASE_MISSION], columns=mission_batch_df_scaled.columns), mission_batch_df_scaled], ignore_index=True)
    
        parameter_data_loader = ParameterDataLoader(
            mission_batch_df=mission_batch_df_scaled,
            mission_batch_with_base_df=df_missions_batch_with_base, 
            mission_batch_travel_df=df_edges,
            fork_lifts_df=df_ops,
            pallet_types_df=df_pallet_types,
            Big_M=BIG_M
        )

        #extract time dicts
        #dict: {(op_id, mission_id): processing_time}
        processing_times = parameter_data_loader.get_mission_processing_times()
        
        #dict: {(mission_id_1, mission_id_2): travel_time}
        travel_times = parameter_data_loader.get_mission_travel_times()

        #initialize HeteroData
        data = HeteroData()
        data['order'].x = x_missions
        data['operator'].x = x_ops
        data['order'].global_id = torch.tensor(mission_ids, dtype=torch.long)
        data['operator'].global_id = torch.tensor(op_ids, dtype=torch.long)

        #global state vector (u) for Meta-Layer
        #[Alpha, Beta, H_fixed] is stored as a global graph feature
        data.u = torch.tensor([[alpha, beta, h_fixed]], dtype=torch.float)

        #-STR-edge feature engineering
        
        #order-order edges (all possible sequencing/scheduling between orders w.r.t. travel times)
        src_list, dst_list, travel_time_list = [], [], []
    
        for (m1, m2), t_time in travel_times.items():
            if m1 in order_map and m2 in order_map:
                src_list.append(order_map[m1])
                dst_list.append(order_map[m2])
                travel_time_list.append([t_time])
        
        edge_index_ord = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr_ord = torch.tensor(travel_time_list, dtype=torch.float)
        
        #normalize travel times (max-normalization)
        #for deep GNNs with attention mechanisms, normalization is mandatory to prevent numerical instability
        if edge_attr_ord.shape[0] > 0:
            edge_attr_ord = edge_attr_ord / edge_attr_ord.max()
        
        data['order', 'to', 'order'].edge_index = edge_index_ord
        data['order', 'to', 'order'].edge_attr = edge_attr_ord

        #operator-order edges (all possible assignment between orders/operators w.r.t. processing times)
        src_ops, dst_ords, proc_time_list = [], [], []
        
        for (op_id, ord_id), p_time in processing_times.items():
            if op_id in op_map and ord_id in order_map:
                src_ops.append(op_map[op_id])
                dst_ords.append(order_map[ord_id])
                proc_time_list.append([p_time])
        
        data['operator', 'assign', 'order'].edge_index = torch.tensor([src_ops, dst_ords], dtype=torch.long)

        #normalize processing times (max-normalization)
        #for deep GNNs with attention mechanisms, normalization is mandatory to prevent numerical instability
        op_edge_attr = torch.tensor(proc_time_list, dtype=torch.float)
        if op_edge_attr.shape[0] > 0:
            op_edge_attr = op_edge_attr / op_edge_attr.max()
        
        data['operator', 'assign', 'order'].edge_attr = op_edge_attr

        #reverse Edge (Order -> Op):  An operator also needs to know about the Orders it might take (to update its own state/embedding)
        rev_edge_index = data['operator', 'assign', 'order'].edge_index.flip([0])
        
        #same edge attributes (travel time is undirected/symmetric usually)
        rev_edge_attr = data['operator', 'assign', 'order'].edge_attr

        data['order', 'rev_assign', 'operator'].edge_index = rev_edge_index
        data['order', 'rev_assign', 'operator'].edge_attr = rev_edge_attr
        

        #-STR-ground truth labelling (based on MIP mini-batch schedule)
        
        #activation Labels
        active_op_indices = set()
        #verify operators that actually exist in Op files
        valid_ops_in_schedule = [o for o in df_schedule['Operator'].unique() if o in op_map]
        
        for op_id in valid_ops_in_schedule:
            active_op_indices.add(op_map[op_id])
                
        y_activation = torch.zeros(num_ops, dtype=torch.float)
        y_activation[list(active_op_indices)] = 1.0
        data['operator'].y = y_activation

        #assignment & sequence labels
        num_op_edges = data['operator', 'assign', 'order'].edge_index.shape[1]
        num_ord_edges = data['order', 'to', 'order'].edge_index.shape[1]
        
        y_assign = torch.zeros(num_op_edges, dtype=torch.float)
        y_seq = torch.zeros(num_ord_edges, dtype=torch.float)
        
        #lookup maps for edge indices
        op_edge_lookup = {
            (s.item(), d.item()): i 
            for i, (s, d) in enumerate(zip(*data['operator', 'assign', 'order'].edge_index))
        }
        ord_edge_lookup = {
            (s.item(), d.item()): i 
            for i, (s, d) in enumerate(zip(*data['order', 'to', 'order'].edge_index))
        }

        for op_id, group in df_schedule.groupby('Operator'):
            if op_id not in op_map: continue
            op_idx = op_map[op_id]
            
            #sort missions by start time
            missions = group.sort_values('Start')
            
            #first mission assignment (op -assign-> first order)
            if not missions.empty:
                first_mission = missions.iloc[0]['Task']
                if first_mission in order_map:
                    f_idx = order_map[first_mission]
                    if (op_idx, f_idx) in op_edge_lookup:
                        y_assign[op_edge_lookup[(op_idx, f_idx)]] = 1.0

            #sequence (order -to-> succ_order)
            for _, row in group.iterrows():
                curr = row['Task']
                succ = row['Successor']
                
                #ignore '0' that denotes end of route (return to base)
                if curr in order_map and succ != 0 and succ in order_map:
                    u, v = order_map[curr], order_map[succ]
                    if (u, v) in ord_edge_lookup:
                        y_seq[ord_edge_lookup[(u, v)]] = 1.0

        data['operator', 'assign', 'order'].y = y_assign
        data['order', 'to', 'order'].y = y_seq

        return data

    # def build_constrained_gnn_data(
    #     mission_file, 
    #     forklift_file,
    #     schedule_file, 
    #     pallet_types_file,
    #     travel_file
    # ):
    #     df_missions = pd.read_csv(mission_file)
    #     df_forklifts = pd.read_csv(forklift_file)
    #     df_edges = pd.read_csv(travel_file)
    #     df_schedule = pd.read_csv(schedule_file)
    #     df_pallet_types = pd.read_csv(pallet_types_file)
        
    #     #extract Global Parameters from schedule_file filename
    #     pattern = r"A(?P<A>[\d.]+)_B(?P<B>[\d.]+)_H(?P<H>\d+)"
    #     match = re.search(pattern, schedule_file)

    #     assert match, f"Can't extract global params from '{schedule_file}'"

    #     alpha = match.group('A')
    #     beta = match.group('B')
    #     h_fixed = match.group('H')
        
    #     data = HeteroData()

    #     #mission nodes
    #     mission_feats = []
    #     for _, row in df_missions.iterrows():
    #         w, l = df_pallet_types.get(row['TP_UDC'], (0, 0))

    #         #feature vector: pysical requirements of the task
    #         mission_feats.append([
    #             row['TO_X'], row['TO_Y'], row['TO_Z'], 
    #             row['WEIGHT'], row['HEIGHT'], w, l
    #         ])
    #     data['mission'].x = torch.tensor(mission_feats, dtype=torch.float)

    #     # --- 2. Forklift Nodes (Capabilities) ---
    #     forklift_ids = sorted(df_forklifts.keys())
    #     f_map = {fid: i for i, fid in enumerate(forklift_ids)}
        
    #     f_feats = [[s['speeds'][0], s['speeds'][1], s['speeds'][2]] for s in df_forklifts.values()]
    #     data['forklift'].x = torch.tensor(f_feats, dtype=torch.float)

    #     # --- 3. Edge Logic: Compatibility & Constraint Matching ---
    #     edge_index, edge_attr, y_label = [], [], []

    #     for f_id, f_info in df_forklifts.items():
    #         for o_idx, o_row in df_missions.iterrows():
    #             w, l = df_pallet_types.get(o_row['TP_UDC'], (0, 0))
                
    #             # Retrieve the specific Skill Score for this Forklift + Pallet Type
    #             # This handles the width/length dependency mentioned
    #             skill_score = f_info['skill_map'].get((w, l), 0)
                
    #             # Only create an edge if the forklift is physically capable (Constraint)
    #             # 1. Skill check (Is score > 0?)
    #             # 2. Weight check (Forklift capacity)
    #             # 3. Height check (Mast height)
    #             if skill_score > 0 and f_info['max_weight'] >= o_row['WEIGHT']:
    #                 f_idx = f_map[f_id]
    #                 edge_index.append([f_idx, o_idx])
                    
    #                 # Pre-calculate the kinematic time based on specific forklift speeds
    #                 k_time = calculate_3d_time(o_row, f_info['speeds'])
                    
    #                 # The GNN sees both the skill (quality) and time (cost)
    #                 edge_attr.append([skill_score, k_time])
                    
    #                 # Label from the MIP solution
    #                 is_assigned = 1.0 if (df_schedule[o_row['CD_MISSION']] == f_id) else 0.0
    #                 y_label.append(is_assigned)

    #     data['forklift', 'assigned_to', 'mission'].edge_index = torch.tensor(edge_index).t().long()
    #     data['forklift', 'assigned_to', 'mission'].edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    #     data['forklift', 'assigned_to', 'mission'].y = torch.tensor(y_label, dtype=torch.float)

    #     return data     

        # u = torch.tensor([[h_fixed, alpha, beta]], dtype=torch.float)

        # #node mapping (CD_MISSION -> graph idx)
        # #ensure that '0' (the base) is included.
        # unique_missions = sorted(df_nodes['CD_MISSION'].unique().tolist())
        # if 0 not in unique_missions:
        #     unique_missions.insert(0, 0) #prepend base at index 0
            
        # mission_to_idx = {m_id: i for i, m_id in enumerate(unique_missions)}
        # num_nodes = len(unique_missions)
        
        # print(f"Graph has {num_nodes} nodes (including Base).")

        # #node features (x)
        # #feature vector: [From_X, From_Y, To_X, To_Y, Processing_Time, Is_Base]
        # #we need to normalize coordinates to [0, 1] for the GNN to learn effectively.
        
        # #create a lookup for node attributes
        # node_lookup = df_nodes.set_index('CD_MISSION')[['FROM_X', 'FROM_Y', 'FROM_Z', 'TO_X', 'TO_Y', 'TO_Z']].to_dict('index')
        
        # #fit Min/Max scaler on all coordinates
        # all_coords = df_nodes[['FROM_X', 'FROM_Y', 'FROM_Z', 'TO_X', 'TO_Y', 'TO_Z']].values
        # scaler = MinMaxScaler()
        # scaler.fit(all_coords)
        
        # feature_list = []
        
        # for m_id in unique_missions:
        #     if m_id == 0:
        #         #base Features: assume Base is at (0,0) or derived from edge file
        #         #for this script, we assume Base is at (0,0) and has 0 processing time
        #         norm_coords = [0.0, 0.0, 0.0, 0.0] 
        #         proc_time = 0.0
        #         is_base = 1.0
        #     else:
        #         #node features
        #         data = node_lookup[m_id]
                
        #         #normalize coordinates
        #         raw_coords = [[data['FROM_X'], data['FROM_Y'], data['TO_X'], data['TO_Y']]]
        #         n_c = scaler.transform(raw_coords)[0] #normalized [x1, y1, x2, y2]
                
        #         norm_coords = n_c.tolist()
        #         proc_time = float(data['DIFF']) #'DIFF' seems to be processing time in your file
        #         is_base = 0.0
                
        #     feature_list.append(norm_coords + [proc_time, is_base])
            
        # x = torch.tensor(feature_list, dtype=torch.float)

        # # 4. Build Edges & Labels
        # src_nodes = []
        # dst_nodes = []
        # edge_attrs = []
        # edge_labels = []
        
        # # A. Identify "Positive" Edges (The ones taken in the schedule)
        # # We use the schedule file to find pairs (Task -> Successor)
        # positive_edges = set()
        # for _, row in df_schedule.iterrows():
        #     task = int(row['Task'])
        #     succ = int(row['Successor'])
            
        #     # In your schedule, 'Task' is the current node, 'Successor' is the next.
        #     # Check if successor is valid (sometimes it might be end of list)
        #     if task in mission_to_idx and succ in mission_to_idx:
        #         u_idx = mission_to_idx[task]
        #         v_idx = mission_to_idx[succ]
        #         positive_edges.add((u_idx, v_idx))
                
        #         # Note: Your schedule implies Task -> Successor. 
        #         # You also need Base -> First Task.
        #         # The 'travel' file is often better for this as it lists every hop explicitly.
                
        # # Alternative: Using the 'travel.csv' which explicitly lists "From_Node -> To_Node"
        # df_optimal = pd.read_csv(schedule_travel_file)
        # for _, row in df_optimal.iterrows():
        #     u_node = int(row['From_Node'])
        #     v_node = int(row['To_Node'])
        #     if u_node in mission_to_idx and v_node in mission_to_idx:
        #         positive_edges.add((mission_to_idx[u_node], mission_to_idx[v_node]))

        # # B. Build the Full Graph (All possible edges from your distance file)
        # # Normalize distances
        # max_dist = df_edges['DISTANCE'].max()
        
        # for _, row in df_edges.iterrows():
        #     m1, m2 = int(row['CD_MISSION_1']), int(row['CD_MISSION_2'])
        #     dist = float(row['DISTANCE'])
            
        #     if m1 in mission_to_idx and m2 in mission_to_idx:
        #         u_idx = mission_to_idx[m1]
        #         v_idx = mission_to_idx[m2]
                
        #         src_nodes.append(u_idx)
        #         dst_nodes.append(v_idx)
                
        #         # Feature: Normalized Distance
        #         edge_attrs.append([dist / max_dist])
                
        #         # Label: 1.0 if this exact edge was used in the optimal schedule
        #         lbl = 1.0 if (u_idx, v_idx) in positive_edges else 0.0
        #         edge_labels.append(lbl)
                
        # edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        # edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        # y = torch.tensor(edge_labels, dtype=torch.float)

        # #create Data Object for the gnn model
        # data = HeteroData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
        
        return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mini_batch_number = 1
schedulePattern = re.compile(fr'schedule.+_{mini_batch_number}.+H\d{{1,4}}.csv')
scheduleTravelPattern = re.compile(fr'schedule.+_{mini_batch_number}.+H\d{{1,4}}_travel.csv')

f_nodes_filename = MISSION_BATCH_DIR.replace('.csv', f'_{mini_batch_number}.csv')
f_edges_filename = MISSION_BATCH_TRAVEL_DIR.replace('.csv', f'_{mini_batch_number}.csv')
f_sched_filename = os.path.join(SCHEDULE_DIR, [f for f in os.listdir(SCHEDULE_DIR) if schedulePattern.match(f)][0])
f_sched_travel_filename = os.path.join(SCHEDULE_DIR, [f for f in os.listdir(SCHEDULE_DIR) if scheduleTravelPattern.match(f)][0])

gnn_data_loader = GnnDataLoader()
# gnn_data = gnn_data_loader.build_constrained_gnn_data(f_nodes_filename,
#                                                    f_edges_filename,
#                                                    f_sched_filename, 
#                                                    f_sched_travel_filename)

data = gnn_data_loader.load_and_process_data(f_nodes_filename,
                                            UDC_TYPES_DIR,
                                            FORK_LIFTS_DIR,
                                            f_edges_filename,
                                            f_sched_filename)
data.to(device)

print(f"\n--- mini-batch [{mini_batch_number}] Generated HeteroData Object ---")
print(data)
print(f"Global Context (Alpha, Beta, H): {data.u}")
print(f"Order Nodes: {data['order'].x.shape}")
print(f"Operator Nodes: {data['operator'].x.shape}")
print(f"Order-Order Edges (Travel Time): {data['order', 'to', 'order'].edge_attr.shape}")
print(f"Op-Order Edges (Processing Time): {data['operator', 'assign', 'order'].edge_attr.shape}")

# print(f"\n--- mini-batch [{mini_batch_number}] GNN Data Object Created ---")
# print(gnn_data)
# print(f"Nodes features shape: {gnn_data.x.shape}")
# print(f"Edge index shape: {gnn_data.edge_index.shape}")
# print(f"Labels shape: {gnn_data.y.shape}")

model = MultiCriteriaGNNModel(
    metadata=data.metadata(),
    hidden_dim=64,
    num_layers=3,
    heads=4
).to(device)

#forward pass example
out = model(
    data.x_dict, 
    data.edge_index_dict, 
    data.edge_attr_dict,
    data.u
)

print("Forward Pass Successful.")
print(f"Activation Logits: {out['activation']}")
print(f"Assignment Logits: {out['assignment']}")
print(f"Sequence Logits:   {out['sequence']}")
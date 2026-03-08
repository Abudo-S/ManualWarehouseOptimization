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
LARGE_SCALE_BATCH_NAME = "Batch9000M"
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts10W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = f"./schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"

H_FIXED_MINUTES = 60 #480 for base shift 
ALPHA = 1.0 #makespan weight
BETA = 1000.0 #operator activation weight (ex. 1000 = fully oriented to operator activation, 50 = balanced)
BIG_M = 1e5

#virtual base mission for operators to start and end their routes, we need just the id to retrieve the travel time to it.
BASE_ORDER_NODE_ID = 0 #assuming '0' or 0 is our base node ID in travel_times

#define a global time scale (e.g., standard 8-hour shift in minutes)
#It must be consistent across training, validation, and decoding
#used to normalize time features (H_fixed, processing time, travel time) to [0,1] range 
#for better training stability, especially when using attention mechanisms in GNNs that can be sensitive to feature scale.
GLOBAL_TIME_SCALE_FACTOR = 480.0 

class GnnDataInstanceBuilder:
    
    def __init__(self, global_time_scale_factor=GLOBAL_TIME_SCALE_FACTOR):
        self.global_time_scale_factor = global_time_scale_factor

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

        assign_labels = False
        if schedule_file_path:
            assign_labels = True

        #parse global parameters
        filename = os.path.basename(schedule_file_path if schedule_file_path else node_file_path) #node_file_path for large-scale batches with no labels
        alpha, beta, h_fixed = self.parse_filename_params(filename)
        #print(f"Global params extracted from[{filename}]: Alpha={alpha}, Beta={beta}, H_fixed={h_fixed}")

        #load dataframe
        df_missions = pd.read_csv(node_file_path)
        df_pallet_types = pd.read_csv(pallet_types_file_path)
        df_ops = pd.read_csv(operator_file_path)
        df_edges = pd.read_csv(edge_file_path)

        if assign_labels:
            df_schedule = pd.read_csv(schedule_file_path)

        #concatenate pallet type features in mission features
        df_missions = df_missions.merge(df_pallet_types[['TP_UDC', 'WIDTH', 'LENGTH']], on='TP_UDC', how='left')
        
        #node idx mappings
        mission_ids = df_missions['CD_MISSION'].astype(str).str.replace(",", "", regex=False).astype(int).unique()
        order_map = {id: i for i, id in enumerate(mission_ids)}
        num_mission = len(mission_ids)
        
        op_ids = df_ops['OID'].unique()
        op_map = {id: i for i, id in enumerate(op_ids)}
        num_ops = len(op_ids)

        #-STR-node embedding
        #in case of operator node: the final objective to capture differences between operators to output different activation probabilities

        #simple normalized ID embedding
        # op_id_emb = torch.arange(num_ops, dtype=torch.float) / num_ops 
        # op_id_emb = op_id_emb.unsqueeze(1)  # [num_ops, 1]

        #strong positional encoding like transformers (8 dimensions instead of 1)
        num_ops = len(op_ids)
        pos_enc_dim = 8  #match or exceed other features

        pos_encodings = []
        for dim in range(pos_enc_dim):
            #sinusoidal encoding
            angle = torch.arange(num_ops, dtype=torch.float) / (10000 ** (2 * dim / pos_enc_dim))
            pos_encodings.append(torch.sin(angle))
            pos_encodings.append(torch.cos(angle))

        op_pos_enc = torch.stack(pos_encodings[:pos_enc_dim], dim=1)  #[num_ops, 8]

        #actual_assignments = set(zip(df_schedule['Operator_ID'], df_schedule['To_Node']))

        scaler = MinMaxScaler()
        
        #-STR-node feature engineering

        #mission features [pallet dims, dest axes]
        missions_features = [
            'WEIGHT', 'HEIGHT', 'WIDTH', 'LENGTH', 'FROM_X', 'FROM_Y', 'FROM_Z','TO_X', 'TO_Y', 'TO_Z'
        ]

        for missions_feature in missions_features:
            if df_missions[missions_feature].dtype == 'object' or df_missions[missions_feature].dtype == 'string':  #only apply to string/object columns
                df_missions[missions_feature] = df_missions[missions_feature].str.replace(",", "", regex=False)

        mission_feats_raw = df_missions[missions_features].fillna(0).astype(float)
        missions_scaled = scaler.fit_transform(mission_feats_raw)
        x_missions = torch.tensor(missions_scaled, dtype=torch.float)

        #operator features [speed, fork dims]
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
        mission_batch_df_scaled['CD_MISSION'] = df_missions['CD_MISSION'].astype(str).str.replace(',', '', regex=False).astype(int)
        
        ops_scaled = scaler.fit_transform(op_feats_raw)
        x_ops = torch.tensor(ops_scaled, dtype=torch.float)

        #id embedding concatenation (simple normalized ID)
        #x_ops = torch.cat([x_ops, op_id_emb], dim=1)  #now shape [num_ops, 8] instead of [num_ops, 7]

        #positional encoding concatenation (strong positional encoding like transformers, 8 dimensions instead of 1)
        x_ops = torch.cat([x_ops, op_pos_enc], dim=1)  #now shape [num_ops, 15] instead of [num_ops, 7]

        #for now we exclude the base node from the graph (mission_id = 0) since it doesn't have real features and can cause numerical instability.
        #it can't be assigned to multiple operators in the same route as we need, so it doesn't add much value to the learning process.
        BASE_MISSION = [0 for _ in range(len(mission_batch_df_scaled.columns))]
        df_missions_batch_with_base = pd.concat([pd.DataFrame([BASE_MISSION], columns=mission_batch_df_scaled.columns), mission_batch_df_scaled], ignore_index=True)
        #df_missions_batch_with_base = mission_batch_df_scaled.copy() #keep base node out for now
        
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

        #normalize alpha, beta and h_fixed to avoid gradient numerical instability
        alpha = alpha / (alpha + beta)
        beta = beta / (alpha + beta)
        h_fixed = h_fixed / self.global_time_scale_factor 

        #global state vector (u) for Meta-Layer
        #[Alpha, Beta, H_fixed] is stored as a global graph feature
        data.u = torch.tensor([[alpha, beta, h_fixed]], dtype=torch.float)

        #store global time scale factor in data for reference (can be used for denormalization later in decoder if needed)
        data.global_scale_factor = torch.tensor([self.global_time_scale_factor], dtype=torch.float)

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
        
        #for deep GNNs with attention mechanisms, normalization is mandatory to prevent numerical instability
        #max_travel_time = 1.0 #default to avoid div/0 
        if edge_attr_ord.shape[0] > 0:
            #normalize travel times (max-normalization)
            #max_travel_time = edge_attr_ord.max().item()
            #edge_attr_ord = edge_attr_ord / max_travel_time

            #global scale normalization (based on a predefined constant that represents the max expected time in the environment)
            edge_attr_ord = edge_attr_ord / self.global_time_scale_factor
        
        data['order', 'to', 'order'].edge_index = edge_index_ord
        data['order', 'to', 'order'].edge_attr = edge_attr_ord

        #in case of max normalization,
        #store max travel time in edge attributes for reference (possibile denormalization later in decoder)
        #data['order', 'to', 'order'].max_val = torch.tensor([max_travel_time])

        #operator-order edges (all possible assignment between orders/operators w.r.t. processing times)
        src_ops, dst_ords, proc_time_list = [], [], []
        
        for (op_id, ord_id), p_time in processing_times.items():
            if op_id in op_map and ord_id in order_map:
                src_ops.append(op_map[op_id])
                dst_ords.append(order_map[ord_id])
                #proc_time_list.append([p_time]) #doesn't consider base travel time

                #fetch travel time from base (e.g., node 0) to the current order
                base_travel_time = travel_times.get((BASE_ORDER_NODE_ID, ord_id), 0.0) 
                
                #store both processing time and base travel time
                proc_time_list.append([p_time, base_travel_time])
        
        data['operator', 'assign', 'order'].edge_index = torch.tensor([src_ops, dst_ords], dtype=torch.long)

        #for deep GNNs with attention mechanisms, normalization is mandatory to prevent numerical instability
        op_edge_attr = torch.tensor(proc_time_list, dtype=torch.float)
        #max_proc_time = 1.0 #default to avoid div/0
        if op_edge_attr.shape[0] > 0:
            #normalize processing times (max-normalization)
            # max_proc_time = op_edge_attr.max().item()
            # op_edge_attr = op_edge_attr / max_proc_time
            op_edge_attr = op_edge_attr / self.global_time_scale_factor
        
        data['operator', 'assign', 'order'].edge_attr = op_edge_attr

        #in case of max normalization,
        #store max processing time in edge attributes for reference (possible denormalization later in decoder)
        #data['operator', 'assign', 'order'].max_val = torch.tensor([max_proc_time])

        #reverse edge (order -> op): An operator also needs to know about the orders it might take (to update its own state/embedding)
        rev_edge_index = data['operator', 'assign', 'order'].edge_index.flip([0])
        
        #same edge attributes (travel time is undirected/symmetric usually)
        rev_edge_attr = data['operator', 'assign', 'order'].edge_attr

        data['order', 'rev_assign', 'operator'].edge_index = rev_edge_index
        data['order', 'rev_assign', 'operator'].edge_attr = rev_edge_attr
        

        #-STR- conditional ground truth labelling (based on MIP mini-batch schedule)
        #for large-scale batches we don't have labels
        if assign_labels:

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

            #print("Reverse edges created:", len(data['order', 'rev_assign', 'operator'].edge_index[0]))

        #store schedule_id for reference (can be used for comparison/linking back to original data)
        data.schedule_id = filename.replace('.csv', '') 
            
        return data

#try to build just one HetroData from a mini-batch
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mini_batch_number = 1
    schedulePattern = re.compile(fr'schedule.+_{mini_batch_number}.+H\d{{1,4}}.csv')
    scheduleTravelPattern = re.compile(fr'schedule.+_{mini_batch_number}.+H\d{{1,4}}_travel.csv')

    f_nodes_filename = MISSION_BATCH_DIR.replace('.csv', f'_{mini_batch_number}.csv')
    f_edges_filename = MISSION_BATCH_TRAVEL_DIR.replace('.csv', f'_{mini_batch_number}.csv')
    f_sched_filename = os.path.join(SCHEDULE_DIR, [f for f in os.listdir(SCHEDULE_DIR) if schedulePattern.match(f)][0])
    f_sched_travel_filename = os.path.join(SCHEDULE_DIR, [f for f in os.listdir(SCHEDULE_DIR) if scheduleTravelPattern.match(f)][0])

    gnn_data_loader = GnnDataInstanceBuilder()

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
    print(f"Activation Probs: {out['activation']}")
    print(f"Assignment Probs: {out['assignment']}")
    print(f"Sequence Probs: {out['sequence']}")
import os
import glob
import re
import math
import json
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ParameterDataLoader import ParameterDataLoader

LARGE_SCALE_BATCH_NAME = "Batch10000M" #Batch1000M, Batch9000M or Batch10000M
#file paths
LARGE_BATCH_DIR = "./datasets/large-batch/batch/"
REPORT_DIR = f"./reports/{LARGE_SCALE_BATCH_NAME}/large-batch/"
LARGE_BATCH_TRAVEL_DIR = "./datasets/large-batch/travel/"
MISSION_LARGE_BATCH_DIR = "./datasets/large-batch/batch/Batch_1_100M_distanced_A1.0_B1000.0_H90.csv"
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_LARGE_BATCH_TRAVEL_DIR = "./datasets/large-batch/travel/Batch_1_100M_travel_distanced.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts200W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = "./schedules/mini-batch/"
GREEDY_SCHEDULE_OUTPUT_DIR = "./output/greedy_schedules/"
REPORT_FILE_NAME = f"{REPORT_DIR}full_batch_greedy_schedule_generation_time.txt"

BIG_M = 1e5

AUG_SEEDS = {
    '': None,
    'aug_extreme_shift+200X': 42,
    'aug_extreme_shift+200Y': 43,
    'aug_flipped_x-axis': 44,
    'aug_flipped_y-axis': 45,
    'aug_shifted_+20X_+20Y': 46,
    'aug_shifted_+50X_-50Y': 47,
    'aug_shifted_-20X_-20Y': 48,
    'aug_shifted_-50X_+50Y': 49,
    'aug_swapped_x&y': 50
}

class GreedyScheduleGenerator:
    def __init__(self, 
                 large_batch_dir, 
                 large_batch_travel_dir, 
                 operator_file_path, 
                 pallet_types_file_path,
                 output_dir):
        """
        Initializes the generator by discovering large batch test files and their corresponding travel files.
        """
        self.large_batch_dir = large_batch_dir
        self.large_batch_travel_dir = large_batch_travel_dir
        self.operator_file_path = operator_file_path
        self.pallet_types_file_path = pallet_types_file_path
        self.output_dir = output_dir
        self.items = []
        
        #ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        #discover large-scale test batches
        search_pattern = os.path.join(self.large_batch_dir, "*_distanced_*")
        all_batches = sorted(glob.glob(search_pattern))
        
        for batch_path in all_batches:
            filename = os.path.basename(batch_path)
            
            # The regex pattern mimicking GnnScheduleDataset
            # Extracts batch identifier to find corresponding travel file
            match = re.search(r'h_(\d+)_', filename) 
            if match:
                batch_num = match.group(1)
                
                # Deduce the edge file path based on GnnScheduleDataset's naming logic
                edge_filename = filename.split('_A')[0].replace('distanced', 'travel_distanced.csv')
                edge_path = os.path.join(self.large_batch_travel_dir, edge_filename)
                
                # Check for alternative naming if the regex format differs slightly in actual usage
                if not os.path.exists(edge_path):
                    # Fallback pattern as seen in user's attached file "Batch10M_travel_distanced_1-3.csv"
                    base_name = filename.replace('_distanced_', '_travel_distanced_')
                    edge_path = os.path.join(self.large_batch_travel_dir, base_name)

                if os.path.exists(batch_path) and os.path.exists(edge_path):
                    self.items.append({
                        'node': batch_path,
                        'edge': edge_path,
                        'id': batch_num,
                        'filename': filename
                    })
                else:
                    print(f"Warning: Missing corresponding travel edge file for batch {filename}")

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
    
    def generate_all(self):
        """Iterates through all discovered items and runs the greedy schedule generator."""
        print(f"Discovered {len(self.items)} valid batch-travel file pairs. Generating schedules...")
        
        df_ops = pd.read_csv(self.operator_file_path)
        df_pallet_types = pd.read_csv(self.pallet_types_file_path) if self.pallet_types_file_path else None

        if os.path.exists(REPORT_FILE_NAME):
            #if the report file already exists, remove it to avoid appending to old data
            os.remove(REPORT_FILE_NAME)

        for item in self.items:
            alpha, beta, h_fixed = self.parse_filename_params(item['filename']) 
            
            print(f"Processing Batch {item['id']} -> {item['filename']}")
            
            for seed_alias, random_seed in AUG_SEEDS.items():  # Generate one deterministic schedule and two with different random seeds
                start = time.perf_counter()

                self._generate_greedy_schedule(
                    batch_file=item['node'],
                    travel_file=item['edge'],
                    df_ops=df_ops,
                    df_pallet_types=df_pallet_types,
                    file_name=item['filename'],
                    horizon=h_fixed,
                    batch_id=item['id'],
                    random_seed=random_seed,
                    seed_alias=seed_alias
                )

                end = time.perf_counter()
                execution_time_seconds = end - start
                schedule_file_name = item['filename'].replace('Batch', 'greedy_schedule').replace('.csv', '' if random_seed is None else f'_{seed_alias}')

                with open(REPORT_FILE_NAME, 'a') as file:
                    file.write(f"{schedule_file_name}: {execution_time_seconds} seconds\n")

    def _generate_greedy_schedule(self,
                                  batch_file,
                                  travel_file, 
                                  df_ops, df_pallet_types, 
                                  file_name, 
                                  horizon, 
                                  batch_id, 
                                  random_seed=None, 
                                  seed_alias=None):
        """
            Core logic to generate greedy assignment while minimizing
            local makespan and activations heuristically, utilizing ParameterDataLoader.
        """
        
        mission_batch_df = pd.read_csv(batch_file).dropna(subset=['CD_MISSION'])

        if random_seed is not None:
            mission_batch_df = mission_batch_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        mission_batch_travel_df = pd.read_csv(travel_file)
        mission_batch_travel_df = mission_batch_travel_df.dropna(subset=['CD_MISSION_1', 'CD_MISSION_2'])
        
        # if '30_500M' in batch_file:
        #     print(f"Processing Batch {batch_file}")
        #     for row_num, mission_id in enumerate(mission_batch_df['CD_MISSION'], start=1):
        #         print(f"Row {row_num}: {mission_id}")

        #clean ids strictly in the dataframe
        mission_batch_df['CD_MISSION'] = mission_batch_df['CD_MISSION'].astype(str).str.replace(',', '').astype(int)
        mission_batch_travel_df['CD_MISSION_1'] = mission_batch_travel_df['CD_MISSION_1'].astype(str).str.replace(',', '').astype(int)
        mission_batch_travel_df['CD_MISSION_2'] = mission_batch_travel_df['CD_MISSION_2'].astype(str).str.replace(',', '').astype(float)
        
        features_to_scale = ['FROM_Z','TO_Z']
        df_to_scale = mission_batch_df[features_to_scale]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_to_scale)

        df_scaled_features = pd.DataFrame(
            scaled_data,
            columns=features_to_scale,
            index=mission_batch_df.index
        )

        df_scaled_features = df_scaled_features.clip(lower=0)
        #df_scaled_features.head()

        df_unscaled_features = mission_batch_df.drop(columns=features_to_scale)

        mission_batch_df_scaled = pd.concat([df_unscaled_features, df_scaled_features], axis=1)
        mission_batch_df_scaled['CD_MISSION'] = mission_batch_df['CD_MISSION'].astype(str).str.replace('.', ',').replace(',', '').astype(int)
        
        missions = mission_batch_df['CD_MISSION'].tolist()
        BASE_MISSION = [0 for _ in range(len(mission_batch_df_scaled.columns))]
        df_missions_batch_with_base = pd.concat([pd.DataFrame([BASE_MISSION], columns=mission_batch_df_scaled.columns), mission_batch_df_scaled], ignore_index=True)

        available_operators = df_ops.iloc[:, 0].astype(str).tolist()
        #we suppose that in real-world scenarios, the number of available operators is limited and known (likely to the upper bound).
        #So we can set a reasonable upper bound for the greedy algorithm to explore.
        max_operators = len(available_operators)
        
        parameter_data_loader = ParameterDataLoader(
            mission_batch_df=mission_batch_df,
            mission_batch_with_base_df=df_missions_batch_with_base,
            mission_batch_travel_df=mission_batch_travel_df,
            fork_lifts_df=df_ops,
            pallet_types_df=df_pallet_types,
            Big_M=BIG_M 
        )
        
        #extract raw times
        raw_proc_times = parameter_data_loader.get_mission_processing_times()
        raw_travel_times = parameter_data_loader.get_mission_travel_times()

        proc_times = {}
        for k, v in raw_proc_times.items():
            if isinstance(k, tuple):
                #format (mission_id, operator_id) as pure integers
                proc_times[(int(float(k[0])), int(float(k[1])))] = float(v)
            else:
                proc_times[int(float(k))] = float(v)

        travel_times = {}
        for k, v in raw_travel_times.items():
            if isinstance(k, tuple):
                #format (from_mission, to_mission) as pure integers
                travel_times[(int(float(k[0])), int(float(k[1])))] = float(v)
            else:
                travel_times[int(float(k))] = float(v)

        #check tuple structures after normalization
        is_proc_tuple = isinstance(list(proc_times.keys())[0], tuple) if proc_times else False
        is_travel_tuple = isinstance(list(travel_times.keys())[0], tuple) if travel_times else False
        
        #calculate minimum processing bounds
        total_processing_time = 0.0
        for mission in missions:
            mission_key = int(mission)
            
            min_p_time = float('inf')
            for op in available_operators:
                op_key = int(op)
                
                if is_proc_tuple:
                    p_time = proc_times.get((op_key, mission_key), proc_times.get(mission_key, 0.0))
                else:
                    p_time = proc_times.get(mission_key, 0.0)
                    
                if p_time < min_p_time:
                    min_p_time = p_time
            
            total_processing_time += (min_p_time if min_p_time != float('inf') else 0.0)

        #calculate average travel time
        all_travel_values = [v for v in travel_times.values() if not math.isnan(v)]
        avg_travel_time = sum(all_travel_values) / len(all_travel_values) if all_travel_values else 0.0
        
        #add total expected travel bounds to processing bounds
        total_estimated_time = total_processing_time + (len(missions) * avg_travel_time)

        #calculate theoretical minimum operators
        num_operators = math.ceil(total_estimated_time / horizon)
        num_operators = max(1, min(num_operators, max_operators))
        
        final_ops_state = None
        
        #increase operators incrementally until horizon is met or max operators is reached
        valid_schedule = True
        while num_operators <= max_operators:
            ops = [{'id': available_operators[i], 'time': 0.0, 'last_mission': None, 'schedule': []} 
                   for i in range(num_operators)]
            
            valid_schedule = True
            
            for mission in missions:
                mission_key = int(mission)
                
                best_op = None
                best_completion_time = float('inf')
                best_travel_time = 0.0
                best_proc_time = 0.0
                
                for op in ops:
                    op_key = int(op['id'])
                    
                    if is_proc_tuple:
                        p_time = proc_times.get((op_key, mission_key), proc_times.get(mission_key, 0.0))
                    else:
                        p_time = proc_times.get(mission_key, 0.0)
                    
                    t_time = 0.0
                    if op['last_mission'] is not None:
                        last_mission_key = int(op['last_mission'])
                        
                        if is_travel_tuple:
                            t_time = travel_times.get((last_mission_key, mission_key), 0.0)
                        else:
                            t_time = travel_times.get(mission_key, 0.0) #fallback if flat
                    
                    completion_time = op['time'] + t_time + p_time
                    
                    if completion_time < best_completion_time:
                        best_completion_time = completion_time
                        best_op = op
                        best_travel_time = t_time
                        best_proc_time = p_time
                
                if best_completion_time > horizon:
                    valid_schedule = False
                    break
                    
                best_op['schedule'].append({
                    'mission_id': mission_key,
                    'start_time': round(best_op['time'] + best_travel_time, 2),
                    'finish_time': round(best_completion_time, 2),
                    'processing_duration': round(best_proc_time, 2),
                    'travel_duration': round(best_travel_time, 2)
                })
                best_op['time'] = best_completion_time
                best_op['last_mission'] = mission_key
                
            if valid_schedule:
                final_ops_state = ops
                break
                
            num_operators += 1
            
        if final_ops_state is None:
            print(f"  Warning: Horizon {horizon} exceeded even with max operators.")
            final_ops_state = ops 
        
        #generate json structural export
        formatted_operators = []
        horizon_violations = []
        
        for op in final_ops_state:
            route_sequence = []
            schedule_length = len(op['schedule'])
            
            for idx, task in enumerate(op['schedule']):
                successor_id = None
                if idx + 1 < schedule_length:
                    successor_id = op['schedule'][idx + 1]['mission_id']
                    
                route_sequence.append({
                    'mission_id': task['mission_id'],
                    'start_time': task['start_time'] if len(route_sequence) > 0 else travel_times.get((0, task['mission_id']), avg_travel_time),
                    'finish_time': task['finish_time'],
                    'processing_duration': task['processing_duration'],
                    'travel_duration': task['travel_duration'],
                    'successor': successor_id
                })
                
                if task['finish_time'] > horizon:
                    horizon_violations.append({
                        "operator_id": op['id'],
                        "mission_id": task['mission_id'],
                        "finish_time": task['finish_time']
                    })
            
            if schedule_length > 0:
                formatted_operators.append({
                    'operator_id': op['id'] if str(op['id']).isdigit() else str(op['id']),
                    'assigned_orders_count': schedule_length,
                    'routes': [route_sequence] 
                })
        
        output_data = {
            "metadata": {
                "num_orders": len(missions),
                "num_operators": len(formatted_operators),
                "valid": len(horizon_violations) == 0 and valid_schedule,
                "schedule_id": f"{file_name.replace('Batch', 'schedule')}",
                "horizon_valid": len(horizon_violations) == 0,
                "horizon_violations": horizon_violations
            },
            "operators": formatted_operators
        }
        
        output_file_path = os.path.join(
                                self.output_dir, 
                                f"{file_name.replace('Batch', 'greedy_schedule').replace('.csv', '.json' if random_seed is None else f'_{seed_alias}.json')}"
                            )
        with open(output_file_path, 'w') as f:
            print(f"Generated schedule: {output_file_path}")
            json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    generator = GreedyScheduleGenerator(
        large_batch_dir=LARGE_BATCH_DIR,
        large_batch_travel_dir=LARGE_BATCH_TRAVEL_DIR,
        operator_file_path=FORK_LIFTS_DIR,
        pallet_types_file_path=UDC_TYPES_DIR,
        output_dir=GREEDY_SCHEDULE_OUTPUT_DIR
    )

    generator.generate_all()
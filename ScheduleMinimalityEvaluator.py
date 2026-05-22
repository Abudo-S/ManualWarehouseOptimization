import json
import pandas as pd
import os
import glob
import re
import math
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from ParameterDataLoader import ParameterDataLoader

LARGE_SCALE_BATCH_NAME = "Batch10000M" #Batch1000M, Batch9000M or Batch10000M
TARGET_MINI_BATCH_SIZE = 10 #number of missions per mini-batch
LARGE_BATCH_DIR = "./datasets/large-batch/batch/"
LARGE_BATCH_TRAVEL_DIR = "./datasets/large-batch/travel/"
MISSION_LARGE_BATCH_DIR = "./datasets/large-batch/batch/Batch_1_100M_distanced_A1.0_B1000.0_H90.csv"
MISSION_LARGE_BATCH_TRAVEL_DIR = "./datasets/large-batch/travel/Batch_1_100M_travel_distanced.csv"
LARGE_SCALE_MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}.csv"
PREPROCESSED_BATCH_DIR = f"./preprocessed/{LARGE_SCALE_BATCH_NAME}/Batch{TARGET_MINI_BATCH_SIZE}M_idx.xlsx" #idx to be replaced cluster idx
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts.csv" #50W
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = f"./schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
PREDICTED_SCHEDULE_DIR = f"./predicted_schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
PREDICTED_LARGE_SCHEDULE_DIR = f"./predicted_schedules/large-scale/batch/"
BATCH_SIZE = 32 #nice to be equal to 32 or 64 since we have small mini-batch instances
H_FIXED_EXCEED_TOLERANCE_MIN = 0.0 #allow schedules to tolerate H_fixed exceedance 
BIG_M = 1e5

class ScheduleMinimalityEvaluator:
    def parse_filename_params(self, filename):
        """
        Extracts Global Parameters A (alpha), B (beta), H (H_fixed) from  schedule_file filename.
        pattern: predicted_schedule10M_1_A1.0_B100.0_H90.csv
        """
        pattern = r"A(?P<A>[\d.]+)_B(?P<B>[\d.]+)_H(?P<H>\d+)"
        match = re.search(pattern, filename)

        assert match, f"Can't extract global params from '{filename}'"
        
        alpha = float(match.group('A'))
        beta = float(match.group('B'))
        h_fixed = float(match.group('H'))
        
        return alpha, beta, h_fixed
    
    def __init__(self, 
                 mission_batch_dir=LARGE_BATCH_DIR,
                 mission_batch_travel_dir=LARGE_BATCH_TRAVEL_DIR,
                 fork_lifts_dir=FORK_LIFTS_DIR, 
                 udc_types_dir=UDC_TYPES_DIR, 
                 predicted_schedule_dir=PREDICTED_LARGE_SCHEDULE_DIR,
                 is_mini_batch=False):
        
        self.mission_batch_dir = mission_batch_dir
        self.mission_batch_travel_dir = mission_batch_travel_dir
        self.fork_lifts_dir = fork_lifts_dir
        self.udc_types_dir = udc_types_dir
        self.predicted_schedule_dir = predicted_schedule_dir

        #discover all predicted_schedule files
        #pattern: schedule..._1_A...B...H...0.json
        search_pattern = os.path.join(predicted_schedule_dir, "predicted_schedule*.json")
        all_schedules = sorted(glob.glob(search_pattern))
        
        self.items = []
        
        #load each predicted schedule and its corresponding minimum schedule, along with their global parameters
        for pred_sched_path in all_schedules:
            filename = os.path.basename(pred_sched_path).replace(".json", ".csv")
            alpha, beta, h_fixed = self.parse_filename_params(filename)
            
            #extract batch number (e.g. '1' from 'schedule10M_1_A...')
            match = re.search(r'_(\d+)_', filename) 
            if match:
                batch_num = match.group(1)

            filename = filename.replace('.csv', '')
            filename = filename.split('_aug')[0] #exclude augmentation suffix
            filename = filename.replace('predicted_schedule', 'Batch') + '.csv'

            if is_mini_batch:
                filename = filename.split('_A')[0].replace('_', '_distanced_')
                batch_mission_path = os.path.join(mission_batch_dir,  filename + '.csv')
            else:
                batch_mission_path = os.path.join(mission_batch_dir, filename)
                
            batch_travel_path = os.path.join(mission_batch_travel_dir, filename.split('_A')[0].replace('distanced', 'travel_distanced') + '.csv')

            if os.path.exists(batch_mission_path) and os.path.exists(batch_travel_path):
                self.items.append({
                    'batch_num': batch_num,
                    'predicted_schedule_path': pred_sched_path,
                    'batch_mission_path': batch_mission_path,
                    'batch_travel_path': batch_travel_path,
                    'fork_lifts_path': fork_lifts_dir,
                    'udc_types_path': udc_types_dir,
                    'alpha': alpha,
                    'beta': beta,
                    'h_fixed': h_fixed
                })
            else:
                print(f"Missing mission or travel file for batch {batch_num}")
    
    def find_available_time_horizons(self):
        return sorted(set(item['h_fixed'] for item in self.items))

    def evaluate_total_flow_time_minimality(self, time_horizon=None):
        """
        Evaluates the total_flow_time minimality gap for each predicted batch schedule compared to its corresponding minimum schedule.
        the negative gap means that the predicted schedule is better than minimal schedule (which can never happen),
        but it means that the model has activated more operators than the minimal schedule (so it got a better total_flow_time),
        Minimality gap is converted to percentage for easier interpretation (e.g. 0.05 becomes 5% gap).
        """

        overall_results = []

        for item in self.items:
            if time_horizon is not None and item['h_fixed'] != time_horizon:
                continue

            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            batch_mission_path = item['batch_mission_path']
            batch_travel_path = item['batch_travel_path']
            fork_lifts_path = item['fork_lifts_path']
            udc_types_path = item['udc_types_path']
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']
                
            with open(pred_path) as f:
                pred = json.load(f)

            mission_batch_features = ['CD_MISSION', 'FROM_X', 'FROM_Y', 'TO_X', 'TO_Y', 'FROM_Z', 'TO_Z', 'TP_UDC', 'DISTANCE']
            mission_batch_travel_features = ['CD_MISSION_1', 'CD_MISSION_2', 'FROM_X', 'FROM_Y', 'TO_X', 'TO_Y', 'DISTANCE']
            forklift_features = ['OID', 'FORK_WIDTH', 'FORK_LENGTH', 'SPEED', 'SPEED_WITH_LOAD', 'UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD']
            udc_types_features = ['TP_UDC', 'WIDTH', 'LENGTH']

            fork_lifts_df = pd.read_csv(fork_lifts_path)[forklift_features]
            udc_types_df = pd.read_csv(udc_types_path)[udc_types_features]

            mission_batch_df = pd.read_csv(batch_mission_path)[mission_batch_features]
            #scale only FROM_Z and TO_Z columns
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
            mission_batch_df_scaled = mission_batch_df_scaled.dropna(subset=['CD_MISSION']) #drop missions with missing cd_mission (if any) to avoid issues in mapping and edge creation
            mission_batch_df_scaled['CD_MISSION'] = mission_batch_df_scaled['CD_MISSION'].astype(str).str.replace(',', '', regex=False).astype(int)
            BASE_MISSION = [0 for _ in range(len(mission_batch_df_scaled.columns))]
            df_missions_batch_with_base = pd.concat([pd.DataFrame([BASE_MISSION], columns=mission_batch_df_scaled.columns), mission_batch_df_scaled], ignore_index=True)
            mission_batch_travel_df = pd.read_csv(batch_travel_path)[mission_batch_travel_features]

            parameter_data_loader = ParameterDataLoader(
                mission_batch_df_scaled,
                df_missions_batch_with_base,
                mission_batch_travel_df,
                fork_lifts_df,
                udc_types_df,
                BIG_M
            )

            travel_times=parameter_data_loader.get_mission_travel_times()
            processing_times=parameter_data_loader.get_mission_processing_times()

            mission_travel_times = defaultdict(list)
            mission_processing_times = defaultdict(list)

            #(cd_mission, cd_mission): travel_time
            {mission_travel_times[k[0]].append(travel_time) for k, travel_time in travel_times.items()}
            total_travel_mins = [min(p_time) for mission, p_time in mission_travel_times.items()]

            #(oid_fork_lift, cd_mission): processing_time
            {mission_processing_times[k[1]].append(processing_time) for k, processing_time in processing_times.items()}
            total_processing_mins =[min(p_time) for mission, p_time in mission_processing_times.items()]

            min_total_flow_time = sum(total_travel_mins) + sum(total_processing_mins)
            t_pruned = (min_total_flow_time/h_fixed) * min(total_travel_mins)
            min_total_flow_time = min_total_flow_time - t_pruned

            pred_total_flow_time = 0.0
            for op in pred["operators"]:
                for route in op["routes"]:     
                    if route:          
                        route.sort(key=lambda x: x["finish_time"])
                        #sum of finish times of last missions in each route, as a proxy for total_flow_time (since we don't have the actual schedule structure here)
                        pred_total_flow_time += route[-1]["finish_time"] 

            minimality_gap = (pred_total_flow_time - min_total_flow_time) / min_total_flow_time if min_total_flow_time > 0 else float('inf')

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_total_flow_time': pred_total_flow_time,
                'min_total_flow_time': min_total_flow_time,
                'minimality_gap': round(minimality_gap, 4) * 100 #convert to percentage
            })
            
        return overall_results
    
    def evaluate_activation_minimality(self, total_flow_time_results, time_horizon=None):
        """
        Evaluates the minimality of each batch predicted schedules in terms of number of activations (operators used), compared to the minimal values.
        Minimality gap is converted to percentage for easier interpretation (e.g. 0.05 becomes 5% gap).
        """

        overall_results = []

        for item in self.items:
            if time_horizon is not None and item['h_fixed'] != time_horizon:
                continue

            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            min_total_flow_time = [total_flow_time_result['min_total_flow_time'] for total_flow_time_result in total_flow_time_results if total_flow_time_result['batch_num'] == batch_num][0]
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']

            with open(pred_path) as f:
                pred = json.load(f)

            pred_activation = len(pred["operators"])  #number of operators used in the predicted schedule
    
            min_activation = math.ceil(min_total_flow_time/h_fixed)

            minimality_gap = (pred_activation - min_activation) / min_activation if min_activation > 0 else float('inf')

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_activation': pred_activation,
                'min_activation': min_activation,
                'minimality_gap': round(minimality_gap, 4) * 100 #convert to percentage
            })
            
        return overall_results
    
    def evaluate_makespan_minimality(self, total_flow_time_results, activation_results, time_horizon=None):
        """
        Evaluates the minimality of each batch predicted schedules in terms of number of makespan (operators used), compared to the minimal values.
        Minimality gap is converted to percentage for easier interpretation (e.g. 0.05 becomes 5% gap).
        """

        overall_results = []

        for item in self.items:
            if time_horizon is not None and item['h_fixed'] != time_horizon:
                continue

            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            min_total_flow_time = [total_flow_time_result['min_total_flow_time'] for total_flow_time_result in total_flow_time_results if total_flow_time_result['batch_num'] == batch_num][0]
            min_activation = [activation_result['min_activation'] for activation_result in activation_results if activation_result['batch_num'] == batch_num][0]
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']

            with open(pred_path) as f:
                pred = json.load(f)

            #take the maximum finish time of all routes in all operators
            pred_makespan = max([sorted(route, key=lambda x: x["finish_time"])[-1]["finish_time"]
                                 for op in pred["operators"] 
                                 for route in op["routes"]])

            min_makespan = min_total_flow_time/min_activation

            minimality_gap = (pred_makespan - min_makespan) / min_makespan if min_makespan > 0 else float('inf')

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_makespan': pred_makespan,
                'min_makespan': min_makespan,
                'minimality_gap': round(minimality_gap, 4) * 100 #convert to percentage
            })
            
        return overall_results

    def evaluate_combined_minimality(self, time_horizon=None):
        """
        Combines makespan and activation minimality gaps using the provided normalized alpha and beta weights per batch,
        to compute a single combined minimality gap metric for each batch schedule.
        All minimality gaps are converted to percentages for easier interpretation (e.g. 0.05 becomes 5% gap).
        """

        overall_results = []

        for item in self.items:
            if time_horizon is not None and item['h_fixed'] != time_horizon:
                continue

            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            batch_mission_path = item['batch_mission_path']
            batch_travel_path = item['batch_travel_path']
            fork_lifts_path = item['fork_lifts_path']
            udc_types_path = item['udc_types_path']
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']

            with open(pred_path) as f:
                pred = json.load(f)

            mission_batch_features = ['CD_MISSION', 'FROM_X', 'FROM_Y', 'TO_X', 'TO_Y', 'FROM_Z', 'TO_Z', 'TP_UDC', 'DISTANCE']
            mission_batch_travel_features = ['CD_MISSION_1', 'CD_MISSION_2', 'FROM_X', 'FROM_Y', 'TO_X', 'TO_Y', 'DISTANCE']
            forklift_features = ['OID', 'FORK_WIDTH', 'FORK_LENGTH', 'SPEED', 'SPEED_WITH_LOAD', 'UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD']
            udc_types_features = ['TP_UDC', 'WIDTH', 'LENGTH']

            fork_lifts_df = pd.read_csv(fork_lifts_path)[forklift_features]
            udc_types_df = pd.read_csv(udc_types_path)[udc_types_features]

            mission_batch_df = pd.read_csv(batch_mission_path)[mission_batch_features]
            #scale only FROM_Z and TO_Z columns
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
            mission_batch_df_scaled = mission_batch_df_scaled.dropna(subset=['CD_MISSION']) #drop missions with missing cd_mission (if any) to avoid issues in mapping and edge creation
            mission_batch_df_scaled['CD_MISSION'] = mission_batch_df_scaled['CD_MISSION'].astype(str).str.replace(',', '', regex=False).astype(int)
            BASE_MISSION = [0 for _ in range(len(mission_batch_df_scaled.columns))]
            df_missions_batch_with_base = pd.concat([pd.DataFrame([BASE_MISSION], columns=mission_batch_df_scaled.columns), mission_batch_df_scaled], ignore_index=True)
            mission_batch_travel_df = pd.read_csv(batch_travel_path)[mission_batch_travel_features]

            parameter_data_loader = ParameterDataLoader(
                mission_batch_df_scaled,
                df_missions_batch_with_base,
                mission_batch_travel_df,
                fork_lifts_df,
                udc_types_df,
                BIG_M
            )

            travel_times=parameter_data_loader.get_mission_travel_times()
            processing_times=parameter_data_loader.get_mission_processing_times()

            mission_travel_times = defaultdict(list)
            mission_processing_times = defaultdict(list)

            #(cd_mission, cd_mission): travel_time
            {mission_travel_times[k[0]].append(travel_time) for k, travel_time in travel_times.items()}
            total_travel_mins = [min(t_time) for mission, t_time in mission_travel_times.items()]

            #(oid_fork_lift, cd_mission): processing_time
            {mission_processing_times[k[1]].append(processing_time) for k, processing_time in processing_times.items()}
            total_processing_mins = [min(p_time) for mission, p_time in mission_processing_times.items()]

            min_total_flow_time = sum(total_travel_mins) + sum(total_processing_mins)
            t_pruned = (min_total_flow_time/h_fixed) * min(total_travel_mins)
            min_total_flow_time = min_total_flow_time - t_pruned

            min_activation = math.ceil(min_total_flow_time/h_fixed)
            min_makespan = min_total_flow_time/min_activation

            #take the maximum finish time of all routes in all operators
            pred_makespan = max([sorted(route, key=lambda x: x["finish_time"])[-1]["finish_time"]
                                 for op in pred["operators"] 
                                 for route in op["routes"]])

            pred_activation = len(pred["operators"]) #number of operators used in the predicted schedule

            makespan_min_gap = (pred_makespan - min_makespan) / min_makespan if min_makespan > 0 else float('inf')
            activation_min_gap = (pred_activation - min_activation) / min_activation if min_activation > 0 else float('inf')

            #normalize alpha, beta to sum to 1 for weighting
            alpha = alpha / (alpha + beta)
            beta = beta / (alpha + beta)

            combined_pred_score = alpha * pred_makespan + beta * pred_activation
            combined_min_score = alpha * min_makespan + beta * min_activation
            combined_min_gap = abs(alpha * makespan_min_gap + beta * activation_min_gap)

            # #number of missions in the original batch
            # num_missions = len(mission_batch_df)
            # #number of missions in the predicted schedule
            # num_pred_missions = sum([len(op["routes"][0]) for op in pred["operators"]])

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_makespan': pred_makespan,
                'min_makespan': min_makespan,
                'pred_activations': pred_activation,
                'min_activations': min_activation,
                'combined_pred_score': combined_pred_score,
                'combined_min_score': combined_min_score,
                'makespan_min_gap': round(makespan_min_gap, 4) * 100, #convert to percentage
                'activation_min_gap': round(activation_min_gap, 4) * 100, #convert to percentage
                'combined_min_gap': round(combined_min_gap, 4) * 100 #convert to percentage
            })
            
        return overall_results
    
    def evaluate_coefficient_of_variation_minimality(self, time_horizon=None):
        """
        Evaluates the minimality of each batch's predicted schedules in terms of 
        workload balancing, using the Coefficient of Variation (CV).
        The theoretical minimum CV is 0.0 (perfectly equal workloads).
        Therefore, the absolute minimality gap is simply the predicted CV.
        """

        overall_results = []

        for item in self.items:
            if time_horizon is not None and item['h_fixed'] != time_horizon:
                continue
            
            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']

            with open(pred_path) as f:
                pred = json.load(f)
            
            #take the finish times of all routes in all operators
            pred_finish_times = [route[-1]["finish_time"]
                                    for op in pred["operators"] 
                                    for route in op["routes"]]

            pred_coefficient_variation = (pd.Series(pred_finish_times).std() / pd.Series(pred_finish_times).mean()).item() if len(pred_finish_times) > 1 else 0.0

            #the theoretical minimum cv is 0.0. 
            #using absolute gap: pred_cv - 0.0 = pred_cv
            min_cv = 0.0
            minimality_gap = pred_coefficient_variation - min_cv 

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_cv': round(pred_coefficient_variation, 4),
                'min_cv': min_cv,
                'minimality_gap': round(minimality_gap, 4)
            })

        return overall_results

if __name__ == "__main__":
    #mini-batch
    evaluator = ScheduleMinimalityEvaluator(mission_batch_dir=f"./datasets/{LARGE_SCALE_BATCH_NAME}/batch",
                                            mission_batch_travel_dir=f"./datasets/{LARGE_SCALE_BATCH_NAME}/travel",
                                            predicted_schedule_dir=PREDICTED_SCHEDULE_DIR, 
                                            is_mini_batch=True)
    
    #evaluator = ScheduleMinimalityEvaluator()
    
    total_flow_time_results = evaluator.evaluate_total_flow_time_minimality()
    activation_results = evaluator.evaluate_activation_minimality(total_flow_time_results)
    makespan_results = evaluator.evaluate_makespan_minimality(total_flow_time_results, activation_results)
    combined_results = evaluator.evaluate_combined_minimality()

    #convert results to DataFrames for better visualization
    df_makespan = pd.DataFrame(makespan_results)
    df_activation = pd.DataFrame(activation_results)
    df_combined = pd.DataFrame(combined_results)

    print("Makespan Minimality Results:")
    print(df_makespan)

    print("\nActivation Minimality Results:")
    print(df_activation)

    print("\nCombined Minimality Results:")
    print(df_combined)
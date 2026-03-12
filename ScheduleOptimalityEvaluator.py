import json
import pandas as pd
import os
import glob
import re

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


class ScheduleOptimalityEvaluator:
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
    
    def __init__(self, schedule_dir, predicted_schedule_dir):
        self.schedule_dir = schedule_dir
        self.predicted_schedule_dir = predicted_schedule_dir
        
        #discover all predicted_schedule files
        #pattern: schedule..._1_A...B...H...0.json
        search_pattern = os.path.join(predicted_schedule_dir, "predicted_schedule*0.json")
        all_schedules = sorted(glob.glob(search_pattern))
        
        self.items = []
        
        #load each predicted schedule and its corresponding optimal schedule, along with their global parameters
        for pred_sched_path in all_schedules:
            filename = os.path.basename(pred_sched_path)
            alpha, beta, h_fixed = self.parse_filename_params(filename)
            
            #extract batch number (e.g. '1' from 'schedule10M_1_A...')
            match = re.search(r'_(\d+)_A', filename) 
            if match:
                batch_num = match.group(1)

            truth_schedule_filename = filename.replace("predicted_schedule", "schedule").replace(".json", ".csv")
            opt_schedule_path = os.path.join(schedule_dir, truth_schedule_filename)

            self.items.append({
                'batch_num': batch_num,
                'predicted_schedule_path': pred_sched_path,
                'optimal_schedule_path': opt_schedule_path,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed
            })

    def evaluate_makespan_optimality(self):
        """
        Evaluates the makespan optimality gap for each predicted batch schedule compared to its corresponding optimal schedule.
        the negative gap means that the predicted schedule is better than optimal schedule (which can never happen),
        but it means that the model has activated more operators than the optimal schedule (so it got a better makespan),
        but will be considered as a sign of suboptimality in terms of activation cost.
        Optimality gap is converted to percentage for easier interpretation (e.g. 0.05 becomes 5% gap).
        """

        overall_results = []

        for item in self.items:
            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            opt_path = item['optimal_schedule_path']
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']

            with open(pred_path) as f:
                pred = json.load(f)

            #take the maximum finish time of all routes in all operators
            pred_makespan = max([sorted(route, key=lambda x: x["finish_time"])[-1]["finish_time"]
                                 for op in pred["operators"] 
                                 for route in op["routes"]])


            df_opt = pd.read_csv(opt_path)
            #sum of finish times of last missions for each operator, as a proxy for makespan
            opt_makespan = df_opt.groupby("Operator")["Finish"].max().max()

            # print(f"Evaluating {pred_path} against {opt_path} with alpha={alpha}, beta={beta}, H_fixed={h_fixed}")
            # print(f"Predicted Makespan: {pred_makespan}, Optimal Makespan: {opt_makespan}")
            # print(f"Relative Error for batch [{batch_num}]: {(pred_makespan - opt_makespan) / opt_makespan:.2%}\n")

            optimality_gap = (pred_makespan - opt_makespan) / opt_makespan if opt_makespan > 0 else float('inf')

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_makespan': pred_makespan,
                'opt_makespan': opt_makespan,
                'optimality_gap': round(optimality_gap, 4) * 100 #convert to percentage
            })
            
        return overall_results
    
    def evaluate_activation_optimality(self):
        """
        Evaluates the optimality of each batch predicted schedules in terms of number of activations (operators used), compared to the optimal schedules.
        Optimality gap is converted to percentage for easier interpretation (e.g. 0.05 becomes 5% gap).
        """

        overall_results = []

        for item in self.items:
            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            opt_path = item['optimal_schedule_path']
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']

            with open(pred_path) as f:
                pred = json.load(f)

            pred_activation = len(pred["operators"])  #number of operators used in the predicted schedule
    
            df_opt = pd.read_csv(opt_path)
            opt_activation = len(df_opt["Operator"].unique()) #number of unique operators used in the optimal schedule (assuming operator IDs are sequential and start from 1)

            optimality_gap = (pred_activation - opt_activation) / opt_activation if opt_activation > 0 else float('inf')

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_activation': pred_activation,
                'opt_activation': opt_activation,
                'optimality_gap': round(optimality_gap, 4) * 100 #convert to percentage
            })
            
        return overall_results

    def evaluate_combined_optimality(self):
        """
        Combines makespan and activation optimality gaps using the provided normalized alpha and beta weights per batch,
        to compute a single combined optimality gap metric for each batch schedule.
        All optimality gaps are converted to percentages for easier interpretation (e.g. 0.05 becomes 5% gap).
        """

        overall_results = []

        for item in self.items:
            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            opt_path = item['optimal_schedule_path']
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']

            with open(pred_path) as f:
                pred = json.load(f)

            #take the maximum finish time of all routes in all operators
            pred_makespan = max([sorted(route, key=lambda x: x["finish_time"])[-1]["finish_time"]
                                 for op in pred["operators"] 
                                 for route in op["routes"]])

            pred_activation = len(pred["operators"]) #number of operators used in the predicted schedule

            df_opt = pd.read_csv(opt_path)
            opt_makespan = df_opt.groupby("Operator")["Finish"].max().max() 
            opt_activation = len(df_opt["Operator"].unique()) #number of unique operators used in the optimal schedule

            makespan_opt_gap = (pred_makespan - opt_makespan) / opt_makespan if opt_makespan > 0 else float('inf')
            activation_opt_gap = (pred_activation - opt_activation) / opt_activation if opt_activation > 0 else float('inf')

            #normalize alpha, beta to sum to 1 for weighting
            alpha = alpha / (alpha + beta)
            beta = beta / (alpha + beta)

            combined_pred_score = alpha * pred_makespan + beta * pred_activation
            combined_opt_score = alpha * opt_makespan + beta * opt_activation
            combined_opt_gap = abs(alpha * makespan_opt_gap + beta * activation_opt_gap)

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_makespan': pred_makespan,
                'opt_makespan': opt_makespan,
                'pred_activations': pred_activation,
                'opt_activations': opt_activation,
                'combined_pred_score': combined_pred_score,
                'combined_opt_score': combined_opt_score,
                'makespan_opt_gap': round(makespan_opt_gap, 4) * 100, #convert to percentage
                'activation_opt_gap': round(activation_opt_gap, 4) * 100, #convert to percentage
                'combined_opt_gap': round(combined_opt_gap, 4) * 100 #convert to percentage
            })
            
        return overall_results

    def evaluate_total_flow_time_optimality(self):
        """
        Evaluates the total_flow_time optimality gap for each predicted batch schedule compared to its corresponding optimal schedule.
        Optimality gap is converted to percentage for easier interpretation (e.g. 0.05 becomes 5% gap).
        Note that it's not considered as a part of the combined optimality gap metric, 
        since it's not directly related to makespan or activation. But it measures the total elapsed time of all missions in the schedule.
        """

        overall_results = []

        for item in self.items:
            batch_num = item['batch_num']
            pred_path = item['predicted_schedule_path']
            opt_path = item['optimal_schedule_path']
            alpha = item['alpha']
            beta = item['beta']
            h_fixed = item['h_fixed']

            with open(pred_path) as f:
                pred = json.load(f)

            pred_total_flow_time = 0.0
            for op in pred["operators"]:
                for route in op["routes"]:     
                    if route:          
                        route.sort(key=lambda x: x["finish_time"])
                        #sum of finish times of last missions in each route, as a proxy for total_flow_time (since we don't have the actual schedule structure here)
                        pred_total_flow_time += route[-1]["finish_time"] 

            df_opt = pd.read_csv(opt_path)
            #sum of finish times of last missions for each operator, as a proxy for total_flow_time
            opt_total_flow_time = df_opt.groupby("Operator")["Finish"].max().sum() 

            # print(f"Evaluating {pred_path} against {opt_path} with alpha={alpha}, beta={beta}, H_fixed={h_fixed}")
            # print(f"Predicted total_flow_time: {pred_total_flow_time}, Optimal total_flow_time: {opt_total_flow_time}")
            # print(f"Relative Error for batch [{batch_num}]: {(pred_total_flow_time - opt_total_flow_time) / opt_total_flow_time:.2%}\n")

            optimality_gap = (pred_total_flow_time - opt_total_flow_time) / opt_total_flow_time if opt_total_flow_time > 0 else float('inf')

            overall_results.append({
                'batch_num': batch_num,
                'alpha': alpha,
                'beta': beta,
                'h_fixed': h_fixed,
                'pred_total_flow_time': pred_total_flow_time,
                'opt_total_flow_time': opt_total_flow_time,
                'optimality_gap': round(optimality_gap, 4) * 100 #convert to percentage
            })
            
        return overall_results

if __name__ == "__main__":
    evaluator = ScheduleOptimalityEvaluator(SCHEDULE_DIR, PREDICTED_SCHEDULE_DIR)
    makespan_results = evaluator.evaluate_makespan_optimality()
    activation_results = evaluator.evaluate_activation_optimality()
    combined_results = evaluator.evaluate_combined_optimality()
    total_flow_time_results = evaluator.evaluate_total_flow_time_optimality()

    #convert results to DataFrames for better visualization
    df_makespan = pd.DataFrame(makespan_results)
    df_activation = pd.DataFrame(activation_results)
    df_combined = pd.DataFrame(combined_results)
    df_total_flow_time = pd.DataFrame(total_flow_time_results)

    print("Makespan Optimality Results:")
    print(df_makespan)

    print("\nActivation Optimality Results:")
    print(df_activation)

    print("\nCombined Optimality Results:")
    print(df_combined)

    print("\nTotal Flow Time Optimality Results:")
    print(df_total_flow_time)
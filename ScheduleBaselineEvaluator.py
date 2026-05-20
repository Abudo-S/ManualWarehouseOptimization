import os
import glob
import re
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from ParameterDataLoader import ParameterDataLoader


LARGE_SCALE_BATCH_NAME = "Batch10000M" #Batch1000M, Batch9000M or Batch10000M
TARGET_MINI_BATCH_SIZE = 10 #number of missions per mini-batch
LARGE_SCALE_MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}.csv"
PREPROCESSED_BATCH_DIR = f"./preprocessed/{LARGE_SCALE_BATCH_NAME}/Batch{TARGET_MINI_BATCH_SIZE}M_idx.xlsx" #idx to be replaced cluster idx
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts200W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = f"./schedules/{LARGE_SCALE_BATCH_NAME}/large-scale/"
PREDICTED_SCHEDULE_DIR = f"./predicted_schedules/{LARGE_SCALE_BATCH_NAME}/large-scale/"
PREDICTED_LARGE_SCHEDULE_DIR = f"./predicted_schedules/large-scale/batch/"
GREEDY_SCHEDULE_OUTPUT_DIR = "./output/greedy_schedules/"

BATCH_SIZE = 32 #nice to be equal to 32 or 64 since we have small mini-batch instances
H_FIXED_EXCEED_TOLERANCE_MIN = 0.0 #allow schedules to tolerate H_fixed exceedance 

BIG_M = 1e5

class ScheduleBaselineEvaluator:
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

    def __init__(self, greedy_schedule_dir, predicted_schedule_dir):
        self.greedy_schedule_dir = greedy_schedule_dir
        self.predicted_schedule_dir = predicted_schedule_dir
        
        # discover all predicted_schedule files (large-scale test batches usually end with .json)
        # Assuming the predicted filenames are something like predicted_schedule_batch_1_H60.json
        # or the previous format predicted_schedule10M_1_A...H60.json
        search_pattern = os.path.join(predicted_schedule_dir, "predicted_schedule*.json")
        all_schedules = sorted(glob.glob(search_pattern))
        
        self.items = []
        
        for pred_sched_path in all_schedules:
            filename = os.path.basename(pred_sched_path)
            alpha, beta, h_fixed = self.parse_filename_params(filename)

            #extract batch number based on common patterns
            batch_num = None
            match_standard = re.search(r'schedule_(\d+)_', filename) 
            
            if match_standard:
                batch_num = match_standard.group(1)               

            if batch_num is not None:
                greedy_filename = os.path.basename(pred_sched_path).replace("predicted_schedule", "greedy_schedule")
                greedy_schedule_path = os.path.join(greedy_schedule_dir, greedy_filename)
                
                if os.path.exists(greedy_schedule_path):
                    self.items.append({
                        'batch_num': batch_num,
                        'predicted_schedule_path': pred_sched_path,
                        'greedy_schedule_path': greedy_schedule_path,
                        'h_fixed': h_fixed,
                        'alpha': alpha,
                        'beta': beta
                    })
                else:
                    print(f"Warning: Greedy schedule {greedy_filename} not found for {filename}")

    def _extract_metrics_from_json(self, json_path):
        """
            calculate makespan, active operators, total flow time, cv for schedule
        """

        with open(json_path, 'r') as f:
            data = json.load(f)
            
        isScheduleValid = data.get('metadata', {}).get('valid', False)

        if not isScheduleValid:
            #if the schedule is invalid, we can assign worst-case metrics or skip it
            #we suppose that it shouldn't happen since the original number of available operators should cover the worst-case scenario (upper bound), but we add this just in case.
            return {
                'makespan': float('inf'),
                'active_operators': float('inf'),
                'total_flow_time': float('inf'),
                'cv': float('inf')
            }
        
        operators = data.get('operators', [])
        
        makespans = []
        flow_times = []
        
        for op in operators:
            routes = op.get('routes', [[]])[0]
            if not routes:
                continue
                
            #local makespan is the finish time of the last task
            local_makespan = routes[-1]['finish_time']
            makespans.append(local_makespan)
            
            #flow time is the sum of finish times of all tasks for this operator
            flow_times.extend([task['finish_time'] for task in routes])
            
        global_makespan = max(makespans) if makespans else 0.0
        active_operators = len(makespans)
        total_flow_time = sum(flow_times)
        
        #cv of makespans
        mean_makespan = np.mean(makespans) if makespans else 0.0
        std_makespan = np.std(makespans) if makespans else 0.0
        cv = (std_makespan / mean_makespan) if mean_makespan > 0 else 0.0
        
        return {
            'makespan': global_makespan,
            'active_operators': active_operators,
            'total_flow_time': total_flow_time,
            'cv': cv
        }

    def evaluate_makespan_improvement(self):
        """
           Evaluates how much the GNN model predicted schedule improves makespan over the greedy baseline.
        """

        results = []
        for item in self.items:
            pred_metrics = self._extract_metrics_from_json(item['predicted_schedule_path'])
            greedy_metrics = self._extract_metrics_from_json(item['greedy_schedule_path'])
            
            pred_makespan = pred_metrics['makespan']
            greedy_makespan = greedy_metrics['makespan']
            
            #gap %: negative means GNN is better
            gap = ((pred_makespan - greedy_makespan) / greedy_makespan) * 100 if greedy_makespan > 0 and greedy_makespan != float('inf') else -100
            
            results.append({
                'batch_num': item['batch_num'],
                'predicted_makespan': pred_makespan,
                'greedy_makespan': greedy_makespan,
                'baseline_gap': gap
            })
            
        return results

    def evaluate_activation_improvement(self):
        """
            Evaluates how the GNN operator activation compares to the greedy baseline.
        """

        results = []
        for item in self.items:
            pred_metrics = self._extract_metrics_from_json(item['predicted_schedule_path'])
            greedy_metrics = self._extract_metrics_from_json(item['greedy_schedule_path'])
            
            pred_ops = pred_metrics['active_operators']
            greedy_ops = greedy_metrics['active_operators']
            
            gap = ((pred_ops - greedy_ops) / greedy_ops) * 100 if greedy_ops > 0 and greedy_ops != float('inf') else -100
            
            results.append({
                'batch_num': item['batch_num'],
                'predicted_operators': pred_ops,
                'greedy_operators': greedy_ops,
                'baseline_gap': gap
            })
            
        return results

    def evaluate_total_flow_time_improvement(self):
        """
            Evaluates total flow time (sum of all completion times).
        """

        results = []
        for item in self.items:
            pred_metrics = self._extract_metrics_from_json(item['predicted_schedule_path'])
            greedy_metrics = self._extract_metrics_from_json(item['greedy_schedule_path'])
            
            pred_flow = pred_metrics['total_flow_time']
            greedy_flow = greedy_metrics['total_flow_time']
            
            gap = ((pred_flow - greedy_flow) / greedy_flow) * 100 if greedy_flow > 0 and greedy_flow != float('inf') else -100
            
            results.append({
                'batch_num': item['batch_num'],
                'predicted_flow_time': pred_flow,
                'greedy_flow_time': greedy_flow,
                'baseline_gap': gap
            })
            
        return results

    def evaluate_coefficient_variation_improvement(self):
        """
            Evaluates workload balancing between operators (lower CV is better).
        """

        results = []
        for item in self.items:
            pred_metrics = self._extract_metrics_from_json(item['predicted_schedule_path'])
            greedy_metrics = self._extract_metrics_from_json(item['greedy_schedule_path'])
            
            pred_cv = pred_metrics['cv']
            greedy_cv = greedy_metrics['cv']
            
            gap = ((pred_cv - greedy_cv) / greedy_cv) * 100 if greedy_cv > 0 and greedy_cv != float('inf') else -100
            
            results.append({
                'batch_num': item['batch_num'],
                'predicted_cv': pred_cv,
                'greedy_cv': greedy_cv,
                'baseline_gap': gap
            })
            
        return results

    def evaluate_combined_improvement(self):
        """
            Combines makespan and activation improvement gaps using the normalized alpha and beta weights.
            Produces a single combined gap metric evaluating how much better (or worse) the GNN performs 
            compared to the greedy baseline.
        """
        overall_results = []

        for item in self.items:
            pred_metrics = self._extract_metrics_from_json(item['predicted_schedule_path'])
            greedy_metrics = self._extract_metrics_from_json(item['greedy_schedule_path'])
            
            pred_makespan = pred_metrics['makespan']
            greedy_makespan = greedy_metrics['makespan']
            
            pred_activation = pred_metrics['active_operators']
            greedy_activation = greedy_metrics['active_operators']
            
            makespan_gap = (pred_makespan - greedy_makespan) / greedy_makespan if greedy_makespan > 0 and greedy_makespan != float('inf') else -100
            activation_gap = (pred_activation - greedy_activation) / greedy_activation if greedy_activation > 0 and greedy_activation != float('inf') else -100
            
            #normalize alpha, beta to sum to 1 for weighting
            alpha = alpha / (alpha + beta)
            beta = beta / (alpha + beta)
            
            #combined scores
            combined_pred_score = alpha * pred_makespan + beta * pred_activation
            combined_greedy_score = alpha * greedy_makespan + beta * greedy_activation
            
            #combined gap using absolute weighting logic similar to OptimalityEvaluator
            combined_gap = abs(alpha * makespan_gap + beta * activation_gap)
            
            overall_results.append({
                'batch_num': item['batch_num'],
                'alpha': alpha,
                'beta': beta,
                'h_fixed': item['h_fixed'],
                'pred_makespan': pred_makespan,
                'greedy_makespan': greedy_makespan,
                'pred_activations': pred_activation,
                'greedy_activations': greedy_activation,
                'combined_pred_score': combined_pred_score,
                'combined_greedy_score': combined_greedy_score,
                'makespan_baseline_gap': round(makespan_gap, 4) * 100,
                'activation_baseline_gap': round(activation_gap, 4) * 100,
                'combined_baseline_gap': round(combined_gap, 4) * 100
            })
            
        return overall_results


if __name__ == "__main__":
        
    evaluator = ScheduleBaselineEvaluator(
        greedy_schedule_dir=GREEDY_SCHEDULE_OUTPUT_DIR,
        predicted_schedule_dir=PREDICTED_LARGE_SCHEDULE_DIR
    )

    makespan_results = evaluator.evaluate_makespan_improvement()
    activation_results = evaluator.evaluate_activation_improvement()
    flow_results = evaluator.evaluate_total_flow_time_improvement()
    cv_results = evaluator.evaluate_coefficient_variation_improvement()

    avg_makespan_gap = np.mean([makespan_result['baseline_gap'] for makespan_result in makespan_results])
    avg_activation_gap = np.mean([activation_result['baseline_gap'] for activation_result in activation_results])
    avg_flow_gap = np.mean([flow_result['baseline_gap'] for flow_result in flow_results])
    avg_cv_gap = np.mean([cv_result['baseline_gap'] for cv_result in cv_results])

    print("\n=== Schedule baseline evaluator summary ===")
    print(f"Total batches evaluated: {len(makespan_results)}")
    print("\n---Average improvement over greedy baseline---")
    print(f"Makespan gap: {avg_makespan_gap:.2f}%")
    print(f"Operator activation gap: {avg_activation_gap:.2f}%")
    print(f"TFT gap: {avg_flow_gap:.2f}%")
    print(f"CV gap: {avg_cv_gap:.2f}%")
    print("===========================================\n")
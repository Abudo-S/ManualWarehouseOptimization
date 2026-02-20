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
MAX_ITERATIONS_PER_ORDER = 10 #max attempts to find a feasible operator for an order based on assignment probs (in iterative repair)

class ScheduleOptimalityEvaluator:
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


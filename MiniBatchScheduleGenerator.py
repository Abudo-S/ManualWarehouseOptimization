import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import importlib
import random
from collections import defaultdict
from pyomo.environ import *
from ParameterDataLoader import ParameterDataLoader
from MultiCriteriaMIPModel import MultiCriteriaMIPModel

#when ALPHA is almost equal to BETA, the solver struggles to find an optimal feasible solution & it converges slowly
H_FIXED_MINUTES = 480 #480 for base shift 
ALPHA = 1.0 #makespan weight
BETA = 100.0 #operator activation weight (ex. 1000 = fully oriented to operator activation, 50 = balanced)
BIG_M = 1e5

LARGE_SCALE_BATCH_NAME = "Batch9000M"
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts10W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = f"./schedules/{LARGE_SCALE_BATCH_NAME}/mini-batch/"
BATCH_NAME = MISSION_BATCH_DIR.replace(f'./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch', '').replace('_distanced.csv', '')
BASE_MISSION = [0, 0, 0, 0, 0, 0, 0, 0, 0]  #virtual base mission for operators to start and end their routes

class MiniBatchScheduleGenerator:
    '''
    A copy of "ManualWarehouseOptimization.ipynb" that aims to solve n mini-batches
    '''

    def apply_initial_assertation(self, 
                                  travel_times, 
                                  processing_times,
                                  skill_scores, 
                                  fork_lifts_df,
                                  udc_types_df,
                                  mission_batch_df_scaled):
        
        #necessary assertations to ensure data consistency before optimization

        mission_travel_times = defaultdict(list)
        mission_processing_times = defaultdict(list)
        pallet_type_skill_scores = defaultdict(list)

        #(cd_mission, cd_mission): travel_time
        {mission_travel_times[k[0]].append(travel_time) for k, travel_time in travel_times.items()}
        total_travel_mins = sum([min(p_time) for mission, p_time in mission_travel_times.items()])
        total_travel_maxs = sum([max(p_time) for mission, p_time in mission_travel_times.items()])

        #(oid_fork_lift, cd_mission): processing_time
        {mission_processing_times[k[1]].append(processing_time) for k, processing_time in processing_times.items()}
        total_processing_mins = sum([min(p_time) for mission, p_time in mission_processing_times.items()])
        total_processing_maxs = sum([max(p_time) for mission, p_time in mission_processing_times.items()])

        #(oid_fork_lift, pallet_type): skill_score
        {pallet_type_skill_scores[k[1]].append(skill_score) for k, skill_score in skill_scores.items() if skill_score > 0}

        assert (total_travel_mins + total_processing_mins) <= H_FIXED_MINUTES * len(fork_lifts_df),\
        "Total estimated minimum required time for all missions exceeds total available operator time."

        pallet_types = udc_types_df["TP_UDC"].to_list()
        assert len([1 for mission_pallet_type in mission_batch_df_scaled["TP_UDC"].tolist() if mission_pallet_type not in pallet_types]) == 0,\
        "Each mission's pallet type needs to be in the list of pallet types."

        assert all([len(skill_scores_list) > 0 for skill_scores_list in pallet_type_skill_scores.values()]),\
        "Each pallet type needs to be in at least one operator score."

        return total_travel_maxs, total_processing_maxs

    def calculate_M_time(self, total_travel_maxs, total_processing_maxs):
        
        #calculate M_value dynamically for Big-M_Time used in constraints

        # P_series = pd.Series(processing_times)
        # max_P_per_mission = P_series.groupby(level=1).max()
        # sum_max_P = max_P_per_mission.sum()

        # T_series = pd.Series(travel_times)
        # sum_T = T_series.sum()

        # M_value = sum_max_P + sum_T + 1.0
        #OR use maximum possible time based on maximum travel and processing times
        M_value = total_travel_maxs + total_processing_maxs + 1.0
        print("Calculated M_value for Big-M_Time is: ", M_value)
        return M_value

    def save_schedule(self, instance, batch_name, h_fixed):
        solution_data = []
        for i in instance.I_max:
            try:
                if value(instance.y[i]) == 1: #in case of unfeasible solution, it launches a pyomo error
                    #operator i is active
                    for j in instance.J_prime:
                        for k in instance.J_prime:
                            
                            #apply the j != k filter to avoid pyomo warning about self-loops
                            if j != k:
                                #check if the index (i, j, k) is valid and exists in z
                                if (i, j, k) in instance.z:
                                    
                                    #check for active flow
                                    if value(instance.z[i, j, k]) > 0:
                                        
                                        #add data only for flow between service orders (not flow to/from Base)
                                        if j in instance.J:
                                            solution_data.append({
                                                'Operator': i,
                                                'Task': j,
                                                'Start': value(instance.S[j]),
                                                'Finish': value(instance.C[j]),
                                                'Successor': k
                                            })
            except Exception as _:
                pass

        df_schedule = pd.DataFrame(solution_data)
        if not df_schedule.empty:
            df_schedule = df_schedule.sort_values(by=['Operator', 'Start']).reset_index(drop=True)
            try:
                #save the schedule to a CSV file
                filename = f'{SCHEDULE_DIR}schedule{batch_name}_A{ALPHA}_B{BETA}_H{h_fixed}.csv'
                df_schedule.to_csv(filename, index=False)
                print(f"\nSuccessfully saved schedule to {filename}")
            except Exception as e:
                print(f"Error saving to CSV: {e}")

    def save_schedule_steps(self, instance, batch_name, h_fixed):
        #initialize list to store all route data
        solution_data = []

        #iterate over all potential operators
        for i in instance.I_max:
            try:
                #skip unactivated operators (using tolerance check)
                if value(instance.y[i]) < 0.5:
                    continue 

                #start at the Base node (0)
                current_node = 0  
                is_route_complete = False
                
                max_steps = len(instance.J) + 2 
                steps = 0
                sequence_rank = 1  #to keep track of the order of visits

                #loop until the route returns to the Base (k=0)
                while not is_route_complete and steps < max_steps:
                    steps += 1
                    found_next_step = False
                    
                    #search for the next step (k) starting from current_node
                    for k in instance.J_prime:
                        if current_node == k:
                            continue # Skip self-loops
                        
                        #check if arc (current_node -> k) is active in variable z
                        #we use try/except or direct check if (i, current_node, k) in instance.z
                        if (i, current_node, k) in instance.z and value(instance.z[i, current_node, k]) > 0.5:
                            
                            travel_time = value(instance.T[current_node, k])
                            
                            #common row data
                            row = {
                                'Operator_ID': i,
                                'Sequence_Rank': sequence_rank,
                                'From_Node': current_node,
                                'To_Node': k,
                                'Travel_Time': travel_time,
                                'Start_Service': None,
                                'Processing_Time': None,
                                'Finish_Service': None,
                                'Activity_Type': 'Travel'
                            }

                            if k == 0:
                                #final movement: Return to Base
                                row['Activity_Type'] = 'Return to Base'
                                
                                solution_data.append(row)
                                is_route_complete = True
                                found_next_step = True
                                break #exit the k loop
                            
                            else:
                                #service movement: Visit Task k
                                start_time = value(instance.S[k])
                                finish_time = value(instance.C[k])
                                proc_time = finish_time - start_time
                                
                                #update row with service details
                                row['Activity_Type'] = 'Service'
                                row['Start_Service'] = start_time
                                row['Processing_Time'] = proc_time
                                row['Finish_Service'] = finish_time
                                
                                solution_data.append(row)
                                
                                #move to next node
                                current_node = k
                                sequence_rank += 1
                                found_next_step = True
                                break #exit the k loop

                    if not found_next_step and not is_route_complete:
                        print(f"Error: Route stopped unexpectedly at node {current_node} for operator {i}.")
                        break
            except Exception as _:
                pass

        df_routes = pd.DataFrame(solution_data)

        if not df_routes.empty:
            #reorder columns
            cols = ['Operator_ID', 'Sequence_Rank', 'From_Node', 'To_Node', 'Activity_Type', 
                    'Travel_Time', 'Start_Service', 'Finish_Service', 'Processing_Time']
            df_routes = df_routes[cols]
            
            #sort by Operator and Sequence
            df_routes = df_routes.sort_values(by=['Operator_ID', 'Sequence_Rank'])
            
            print("Preview of Export Data:")
            print(df_routes.head())

            filename = f'{SCHEDULE_DIR}schedule{batch_name}_A{ALPHA}_B{BETA}_H{h_fixed}_travel.csv'
            df_routes.to_csv(filename, index=False)
            print(f"\nSuccessfully saved routes to {filename}")

        else:
            print("No active routes found to export.")


    def __init__(self, n_mini_batches, start_n=1, use_h_fixed=True):

        #with mission TP_UDC, we can retreive relative width and length from mission types
        #mission_batch_features = ['CD_MISSION', 'TP_MISSION', 'FROM_X', 'FROM_Y', 'TO_X', 'TO_Y', 'TP_UDC', 'DISTANCE']
        mission_batch_features = ['CD_MISSION', 'FROM_X', 'FROM_Y', 'TO_X', 'TO_Y', 'FROM_Z', 'TO_Z', 'TP_UDC', 'DISTANCE']
        udc_types_features = ['TP_UDC', 'WIDTH', 'LENGTH']
        mission_batch_travel_features = ['CD_MISSION_1', 'CD_MISSION_2', 'FROM_X', 'FROM_Y', 'TO_X', 'TO_Y', 'DISTANCE']
        fork_lifts_features = ['OID', 'FORK_WIDTH', 'FORK_LENGTH', 'SPEED', 'SPEED_WITH_LOAD', 'UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD']
        #fork_lifts_features = ['OID', 'FORK_WIDTH', 'FORK_LENGTH', 'SPEED', 'SPEED_WITH_LOAD']
        #mission_types_features = ['TP_MISSION', 'DSC_MISSION']
        features_to_scale = ['FROM_Z','TO_Z']
        possibile_h_fixed = [60, 90, 120, 480]

        for mini_batch_number in range(start_n, n_mini_batches + 1):
            print(f'----------------------------WORKING WITH MINI-BATCH [{mini_batch_number}]----------------------------')
            mission_batch_df = pd.read_csv(MISSION_BATCH_DIR.replace('.csv', f'_{mini_batch_number}.csv'))[mission_batch_features]
            udc_types_df = pd.read_csv(UDC_TYPES_DIR)[udc_types_features]
            mission_batch_travel_df = pd.read_csv(MISSION_BATCH_TRAVEL_DIR.replace('.csv', f'_{mini_batch_number}.csv'))[mission_batch_travel_features]
            fork_lifts_df = pd.read_csv(FORK_LIFTS_DIR)[fork_lifts_features]
            #mission_types_df = pd.read_csv(MISSION_TYPES_DIR)[mission_types_features]

            df_to_scale = mission_batch_df[features_to_scale]

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_to_scale)

            df_scaled_features = pd.DataFrame(
                scaled_data,
                columns=features_to_scale,
                index=mission_batch_df.index
            )

            df_scaled_features = df_scaled_features.clip(lower=0)
            df_unscaled_features = mission_batch_df.drop(columns=features_to_scale)
            mission_batch_df_scaled = pd.concat([df_unscaled_features, df_scaled_features], axis=1)

            #in case of merging WIDTH and LENGTH from UDC types into mission batch, so there'd be no need to access UDC types during optimization
            # mission_batch_df_scaled = pd.merge(mission_batch_df_scaled, udc_types_df, on='TP_UDC')
            # mission_batch_df_scaled.drop(columns=['TP_UDC'], inplace=True)

            mission_batch_df_scaled['CD_MISSION'] = mission_batch_df_scaled['CD_MISSION'].str.replace(",", "", regex=False).astype(int)

            mission_batch_df_with_base = pd.concat([pd.DataFrame([BASE_MISSION], columns=mission_batch_df_scaled.columns), mission_batch_df_scaled], ignore_index=True)
            mission_batch_df_scaled['TP_UDC'].fillna(udc_types_df.iloc[0]['TP_UDC'], inplace=True) #the udc_type base mission will remain 0
            mission_batch_df_with_base.head()

            #remove pallet types not present in the mission batch
            udc_types_in_mission_batch = mission_batch_df_scaled['TP_UDC'].unique().tolist()
            udc_types_df = udc_types_df[udc_types_df['TP_UDC'].isin(udc_types_in_mission_batch)]

            parameter_data_loader = ParameterDataLoader(
                mission_batch_df_scaled,
                mission_batch_df_with_base,
                mission_batch_travel_df,
                fork_lifts_df,
                udc_types_df,
                BIG_M
            )

            missions = mission_batch_df_scaled['CD_MISSION'].astype(int).to_list()
            operators = fork_lifts_df['OID'].astype(int).to_list()
            pallet_types = udc_types_df['TP_UDC'].astype(int).to_list()
            missions_with_base = mission_batch_df_with_base['CD_MISSION'].astype(int).to_list()

            travel_times = parameter_data_loader.get_mission_travel_times()
            processing_times = parameter_data_loader.get_mission_processing_times()
            skill_scores = parameter_data_loader.get_operator_skill_scores()
            mission_pallet_types = parameter_data_loader.get_mission_pallet_types()

            print(f'missions: {missions}')
            print(f'missions_with_base: {missions_with_base}')
            print(f'pallet_types: {pallet_types}')
            print(f'operators: {operators}')
            print(f'travel_times: {travel_times}')
            print(f'processing_times: {processing_times}')
            print(f'skill_scores: {skill_scores}')
            print(f'mission_pallet_types: {mission_pallet_types}')

            total_travel_maxs, total_processing_maxs = self.apply_initial_assertation(travel_times, 
                                                                                      processing_times, 
                                                                                      skill_scores, 
                                                                                      fork_lifts_df,
                                                                                      udc_types_df, 
                                                                                      mission_batch_df_scaled)
            M_value = self.calculate_M_time(total_travel_maxs, total_processing_maxs)

            h_fixed = H_FIXED_MINUTES if use_h_fixed else random.choice(possibile_h_fixed)
            mcmModel = MultiCriteriaMIPModel(missions,
                                 operators,
                                 pallet_types,
                                 missions_with_base,
                                 processing_times,
                                 travel_times,
                                 skill_scores,
                                 mission_pallet_types,
                                 h_fixed,
                                 ALPHA,
                                 BETA,
                                 M_value,
                                 BIG_M
                                )
            
            instance, results = mcmModel.solve(solver_name="cplex_direct")
            
            batch_name = f"{BATCH_NAME}_{mini_batch_number}"
            self.save_schedule(instance, batch_name, h_fixed)
            self.save_schedule_steps(instance, batch_name, h_fixed)

if __name__ == "__main__":
    miniBatchScheduleGenerator = MiniBatchScheduleGenerator(n_mini_batches=92, use_h_fixed=False)
import pandas as pd
import pyomo.environ as pyo
import random

#Maximum allowed difference between fork dimensions and pallet type dimensions to consider the fork lift suitable for the pallet type.
FORK_DIMENSIONS_EXCEEDING_THRESHOLD = 0.20 #percentage in fork length/width.
ESTIMATED_TRAVEL_TIME_DELAY_PER_MISSION = 10 #minutes
ESTIMATED_PROCESSING_TIME_DELAY_PER_MISSION = 10 #minutes
ESTIMATED_POSSIBLE_DELAYS = [5, 10, 15, 20] #substitutes fixed delays, it that can be added to travel/processing times to simulate uncertainties.

class ParameterDataLoader:

    def __init__(self, 
                 mission_batch_df:pd.DataFrame, 
                 mission_batch_with_base_df:pd.DataFrame, 
                 mission_batch_travel_df:pd.DataFrame, 
                 fork_lifts_df:pd.DataFrame, 
                 pallet_types_df:pd.DataFrame,
                 Big_M):
        
        self.mission_batch_df = mission_batch_df
        self.mission_batch_with_base_df = mission_batch_with_base_df
        self.mission_batch_travel_df = mission_batch_travel_df[['CD_MISSION_1', 'CD_MISSION_2', 'DISTANCE']]
        self.fork_lifts_df = fork_lifts_df
        self.pallet_types_df = pallet_types_df
        self.Big_M = Big_M #can be used to set a default high value for unknown/unwanted parameters (so they will be neglected by the optimization model)

        #self.fork_lifts_df.columns.str.strip() #remove leading and trailing spaces from column names
        self.fork_lifts_df['SPEED'] = self.fork_lifts_df['SPEED'].fillna(300)
        self.mean_fork_lift_speed = self.fork_lifts_df['SPEED'].mean()

        cols = ['UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD']
        self.fork_lifts_df[cols] = self.fork_lifts_df[cols].fillna(30)
        

    # def get_data_portal(self, model:pyo.AbstractModel) -> pyo.DataPortal:
    #     data_portal = pyo.DataPortal()
    #     mission_batch_list= self.mission_batch_df.CD_MISSION.to_list()
    #     print(f"DEBUG: mission_batch_list is: {mission_batch_list}")
    #     print(f"DEBUG: Type is: {type(mission_batch_list)}")
        
    #     data_portal.load(set=model.J.name, data=mission_batch_list, using=mission_batch_list)

    #     return data_portal

    def get_mission_processing_times(self) -> dict:
        '''
            Returns a dictionary mapping mission codes to their processing times w.r.t. each operator.
            [(operator_id, mission_code)]: processing_time + estimated_delay
            The processing time of a mission is defined as the time taken by an operator to complete the mission.
            A mission processing time is calculated as a sum the time of loading, the time to travel with the relative pallet, and the time of unloading.
        '''

        return {(int(fork_lift['OID']), int(mission['CD_MISSION'])): (mission['FROM_Z']/fork_lift['UP_SPEED']) + 
                                (mission['FROM_Z']/fork_lift['DOWN_SPEED_WITH_LOAD']) + 
                                (mission['DISTANCE']/fork_lift['SPEED_WITH_LOAD']) + 
                                (mission['TO_Z']/fork_lift['UP_SPEED_WITH_LOAD']) +
                                (mission['TO_Z']/fork_lift['DOWN_SPEED']) +
                                ESTIMATED_PROCESSING_TIME_DELAY_PER_MISSION #random.choice(ESTIMATED_POSSIBLE_DELAYS)
                                for forlift_idx, fork_lift in self.fork_lifts_df.iterrows() for mission_idx, mission in self.mission_batch_df.iterrows()}

    def get_mission_travel_times(self) -> dict:
        '''
            Returns a dictionary mapping each sequence of 2 mission codes to their travel times.
            [(mission_code_1, mission_code_2)]: travel_time + estimated_delay
        '''
        cd_missions = self.mission_batch_with_base_df['CD_MISSION'].astype(int).tolist()
        travel_distances = self.mission_batch_travel_df.set_index(['CD_MISSION_1', 'CD_MISSION_2'])['DISTANCE'].to_dict()

        return {(int(k[0]), int(k[1])): (distance / self.mean_fork_lift_speed) + ESTIMATED_TRAVEL_TIME_DELAY_PER_MISSION #random.choice(ESTIMATED_POSSIBLE_DELAYS) 
                for k, distance in travel_distances.items()
                if k[0] in cd_missions and k[1] in cd_missions}
    
    def get_operator_skill_scores(self) -> dict:
        '''
            Returns a dictionary mapping each operator and mission code to a skill score.
            [(operator_id, pallet_type)]: skill_score
            The skill score is a measure of how the fork lift is adequate to pallet type.
            If the pallet dimensions excced forks dimension w.r.t. the threshold, n0 is assigned as operator skill for such pallet type.
            lower the difference between pallet dimensions and fork dimensions, higher the skill score.
        '''

        return {(int(fork_lift['OID']), int(pallet_type['TP_UDC'])):  0
                if (pallet_type['WIDTH'] - fork_lift['FORK_WIDTH']) > (fork_lift['FORK_WIDTH'] + (fork_lift['FORK_WIDTH'] * FORK_DIMENSIONS_EXCEEDING_THRESHOLD)) or \
                (pallet_type['LENGTH'] - fork_lift['FORK_LENGTH']) > (fork_lift['FORK_LENGTH'] + (fork_lift['FORK_LENGTH'] * FORK_DIMENSIONS_EXCEEDING_THRESHOLD))
                else (100 - (abs((pallet_type['WIDTH'] - fork_lift['FORK_WIDTH'])) + abs((pallet_type['LENGTH'] - fork_lift['FORK_LENGTH']))))
                for forlift_idx, fork_lift in self.fork_lifts_df.iterrows() for pallet_type_idx, pallet_type in self.pallet_types_df.iterrows()}
    
    def get_mission_pallet_types(self) -> dict:
        '''
            Returns a dictionary mapping each mission code to its pallet type.
            {mission_code: pallet_type}
        '''
        return {(int(mission['CD_MISSION']), int(pallet_type['TP_UDC'])) : 1 
                if mission['TP_UDC'] == pallet_type['TP_UDC'] 
                else 0
                for mission_idx, mission in self.mission_batch_df.iterrows() for pallet_type_idx, pallet_type in self.pallet_types_df.iterrows()}



import pandas as pd
import pyomo.environ as pyo

class ParameterDataLoader:

    def __init__(self, mission_batch_df:pd.DataFrame, mission_batch_travel_df:pd.DataFrame, fork_lifts_df:pd.DataFrame, Big_M):
        self.mission_batch_df = mission_batch_df
        self.mission_batch_travel_df = mission_batch_travel_df[['CD_MISSION_1', 'CD_MISSION_2', 'DISTANCE']]
        self.fork_lifts_df = fork_lifts_df
        self.Big_M = Big_M #can be used to set default high values for unknown/unwanted parameter's values

        #self.fork_lifts_df.columns.str.strip() #remove leading and trailing spaces from column names
        self.fork_lifts_df['SPEED'].fillna(300, inplace=True)
        self.mean_fork_lift_speed = self.fork_lifts_df['SPEED'].mean()

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
            [(operator_id, mission_code)]: processing_time
            The processing time of a mission is defined as the time taken by an operator to complete the mission.
            A mission processing time is calculated as a sum the time of loading, the time to travel with the relative pallet, and the time of unloading.
        '''
        self.fork_lifts_df[['UP_SPEED', 'UP_SPEED_WITH_LOAD', 'DOWN_SPEED', 'DOWN_SPEED_WITH_LOAD']].fillna(30, inplace=True)
        
        return {(fork_lift['OID'], mission['CD_MISSION']):  (mission['FROM_Z']/fork_lift['UP_SPEED']) + 
                                (mission['FROM_Z']/fork_lift['DOWN_SPEED_WITH_LOAD']) + 
                                (mission['DISTANCE']/fork_lift['SPEED_WITH_LOAD']) + 
                                (mission['TO_Z']/fork_lift['UP_SPEED_WITH_LOAD']) +
                                (mission['TO_Z']/fork_lift['DOWN_SPEED']) 
                                for forlift_idx, fork_lift in self.fork_lifts_df.iterrows() for mission_idx, mission in self.mission_batch_df.iterrows()}

    def get_mission_travel_times(self) -> dict:
        '''
            Returns a dictionary mapping each sequence of 2 mission codes to their travel times.
            [(mission_code_1, mission_code_2)]: travel_time
        '''
       
        travel_distances = self.mission_batch_travel_df.set_index(['CD_MISSION_1', 'CD_MISSION_2'])['DISTANCE'].to_dict()

        return {k: distance / self.mean_fork_lift_speed for k, distance in travel_distances.items()}
    
    def get_operator_skill_scores(self) -> dict:
        #
        return {}



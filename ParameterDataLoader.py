import pandas as pd
import pyomo.environ as pyo

class ParameterDataLoader:

    def __init__(self, mission_batch_df, mission_batch_travel_df, fork_lifts_df):
        self.mission_batch_df = mission_batch_df
        self.mission_batch_travel_df = mission_batch_travel_df[['CD_MISSION_1', 'CD_MISSION_2', 'DISTANCE']]
        self.fork_lifts_df = fork_lifts_df
        self.mean_fork_lift_speed = self.fork_lifts_df.SPEED.mean()

    # def get_data_portal(self, model:pyo.AbstractModel) -> pyo.DataPortal:
    #     data_portal = pyo.DataPortal()
    #     mission_batch_list= self.mission_batch_df.CD_MISSION.to_list()
    #     print(f"DEBUG: mission_batch_list is: {mission_batch_list}")
    #     print(f"DEBUG: Type is: {type(mission_batch_list)}")
        
    #     data_portal.load(set=model.J.name, data=mission_batch_list, using=mission_batch_list)

    #     return data_portal

    def get_mission_processing_times(self) -> dict:
        #
        return {}

    def get_mission_travel_times(self) -> dict:
        travel_distances = self.mission_batch_travel_df.set_index(['CD_MISSION_1', 'CD_MISSION_2'])['DISTANCE'].to_dict()

        return {k: distance / self.mean_fork_lift_speed for k, distance in travel_distances.items()}
    
    def get_operator_skill_scores(self) -> dict:
        #
        return {}



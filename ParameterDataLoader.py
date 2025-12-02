import pandas as pd
from pyomo.environ import *

class ParameterDataLoader:
    def __init__(self, mission_batch_df, mission_batch_travel_df, fork_lifts_df, mission_types_df):
        self.mission_batch_df = mission_batch_df
        self.mission_batch_travel_df = mission_batch_travel_df
        self.fork_lifts_df = fork_lifts_df
        self.mission_types_df = mission_types_df




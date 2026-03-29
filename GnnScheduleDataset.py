import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import os
import glob
import re
import random
from GnnDataInstanceBuilder import GnnDataInstanceBuilder
from MultiCriteriaGNNModel import MultiCriteriaGNNModel


LARGE_SCALE_BATCH_NAME = "Batch1000M" #Batch1000M, Batch9000M or Batch10000M
#file paths
LARGE_BATCH_DIR = "./datasets/large-batch/batch/"
LARGE_BATCH_TRAVEL_DIR = "./datasets/large-batch/travel/"
MISSION_LARGE_BATCH_DIR = "./datasets/large-batch/batch/Batch_1_100M_distanced_A1.0_B1000.0_H90.csv"
MISSION_BATCH_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_LARGE_BATCH_TRAVEL_DIR = "./datasets/large-batch/travel/Batch_1_100M_travel_distanced.csv"
MISSION_BATCH_TRAVEL_DIR = f"./datasets/{LARGE_SCALE_BATCH_NAME}/mini-batch/Batch10M_travel_distanced.csv"
FORK_LIFTS_DIR = "./datasets/ForkLifts10W.csv"
#MISSION_TYPES_DIR = "./datasets/MissionTypes.csv"
SCHEDULE_DIR = "./schedules/mini-batch/"

class GnnScheduleDataset(Dataset):
    """
    torch dataset that discovers all schedule files and pairs them with 
    their corresponding mission/edge CSVs and retrieve their HeteroData instance.
    """
    def __init__(self,
                schedule_dir, 
                mission_base_path, 
                edge_base_path, 
                pallet_types_file_path, 
                fork_path,
                large_batch_dir=None,
                large_batch_travel_dir=None):
        
        self.schedule_dir = schedule_dir
        self.large_batch_dir = large_batch_dir

        self.builder = GnnDataInstanceBuilder()
        self.pallet_types_file_path = pallet_types_file_path
        self.fork_path = fork_path
        
        #discover all schedule files
        #pattern: schedule..._1_A...B...H...0.csv
        if schedule_dir:
            search_pattern = os.path.join(schedule_dir, "schedule*0.csv")
            all_schedules = sorted(glob.glob(search_pattern))
            
            self.items = []
            
            #match each schedule to its batch files
            #assumes filenames like: ..._1_... matches Batch10M_..._1.csv
            for sched_path in all_schedules:
                filename = os.path.basename(sched_path)
                
                #extract batch number (e.g. '1' from 'schedule10M_1_A...')
                match = re.search(r'_(\d+)_A', filename) 
                if match:
                    batch_num = match.group(1)
                    
                    #add corresponding paths
                    node_path = mission_base_path.replace('.csv', f'_{batch_num}.csv')
                    edge_path = edge_base_path.replace('.csv', f'_{batch_num}.csv')
                    
                    if os.path.exists(node_path) and os.path.exists(edge_path):
                        self.items.append({
                            'schedule': sched_path,
                            'node': node_path,
                            'edge': edge_path,
                            'id': batch_num
                        })
                    else:
                        print(f"Warning: Missing node/edge files for schedule {filename}")
        else: #large-scale (test) batches don't have any ground-truth schedules
            search_pattern = os.path.join(large_batch_dir, "*_distanced_*")
            all_batches = sorted(glob.glob(search_pattern))
            
            self.items = []
            
            #match each batch to its travel files
            #assumes filenames like: ..._1 matches Batch_1_100M_distanced_A1.0_B1000.0_H90.csv
            for batch_path in all_batches:
                filename = os.path.basename(batch_path)
                
                #extract batch number (e.g. '1' from 'Batch_1_...')
                match = re.search(r'h_(\d+)_', filename) 
                if match:
                    batch_num = match.group(1)
                    
                    #add corresponding paths
                    node_path = os.path.join(large_batch_dir, filename)
                    edge_path = os.path.join(large_batch_travel_dir, filename.split('_A')[0].replace('distanced', 'travel_distanced.csv'))
                    
                    if os.path.exists(node_path) and os.path.exists(edge_path):
                        self.items.append({
                            'node': node_path,
                            'edge': edge_path,
                            'id': batch_num
                        })
                    else:
                        print(f"Warning: Missing node/edge files for batch {filename}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        #load graph HeteroData instance
        data = self.builder.load_and_process_data(
            node_file_path=item['node'],
            pallet_types_file_path=self.pallet_types_file_path,
            operator_file_path=self.fork_path,
            edge_file_path=item['edge'],
            schedule_file_path=item.get('schedule', None)
        )
        
        return data

    def apply_spatial_augmentation(self, batch):
        """
        Applies distance-preserving spatial augmentations to order coordinates (translation, mirroring and rotation).
        Expects order features where indices 4-9 are: FROM_X, FROM_Y, FROM_Z, TO_X, TO_Y, TO_Z.
        It could be using inside the training loop to change a single batch.
        """
        #print("Applying spatial augmentation to batch...")

        #flip x-axis randomly (50% chance)
        if random.random() > 0.5:
            #assuming origin is 0, we can just invert the sign
            batch['order'].x[:, 4] = -batch['order'].x[:, 4] #FROM_X
            batch['order'].x[:, 7] = -batch['order'].x[:, 7] #TO_X

        #flip y-axis randomly (50% chance)
        if random.random() > 0.5:
            batch['order'].x[:, 5] = -batch['order'].x[:, 5] #FROM_Y
            batch['order'].x[:, 8] = -batch['order'].x[:, 8] #TO_Y

        #random translation (shift coordinates by a random offset)
        #shift the whole layout between -20 and +20 units
        shift_x = random.uniform(-20.0, 20.0)
        shift_y = random.uniform(-20.0, 20.0)
        
        batch['order'].x[:, 4] += shift_x #FROM_X
        batch['order'].x[:, 7] += shift_x #TO_X
        
        batch['order'].x[:, 5] += shift_y #FROM_Y
        batch['order'].x[:, 8] += shift_y #TO_Y
        
        #swap x and y axes randomly (90-degree diagonal rotation equivalent)
        if random.random() > 0.5:
            #swap FROM_X and FROM_Y
            temp_from_x = batch['order'].x[:, 4].clone()
            batch['order'].x[:, 4] = batch['order'].x[:, 5]
            batch['order'].x[:, 5] = temp_from_x
            
            #swap TO_X and TO_Y
            temp_to_x = batch['order'].x[:, 7].clone()
            batch['order'].x[:, 7] = batch['order'].x[:, 8]
            batch['order'].x[:, 8] = temp_to_x

        return batch


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #init mini-batch dataset
    # dataset = GnnScheduleDataset(
    #     schedule_dir=SCHEDULE_DIR,
    #     mission_base_path=MISSION_BATCH_DIR,
    #     edge_base_path=MISSION_BATCH_TRAVEL_DIR,
    #     pallet_types_file_path=UDC_TYPES_DIR,
    #     fork_path=FORK_LIFTS_DIR
    # )

    #init large-batch dataset
    dataset = GnnScheduleDataset(
        schedule_dir=None,
        mission_base_path=MISSION_LARGE_BATCH_DIR,
        edge_base_path=MISSION_LARGE_BATCH_TRAVEL_DIR,
        pallet_types_file_path=UDC_TYPES_DIR,
        fork_path=FORK_LIFTS_DIR,
        large_batch_dir=LARGE_BATCH_DIR,
        large_batch_travel_dir=LARGE_BATCH_TRAVEL_DIR
    )
    
    print(f"Found {len(dataset)} valid schedule instances.")

    #create DataLoader using the dataset
    #batch_size can be > 1 to train on multiple graphs at once
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    #init model
    if len(dataset) > 0:
        sample_data = dataset[0]
        model = MultiCriteriaGNNModel(
            metadata=sample_data.metadata(),
            hidden_dim=64,
            num_layers=3,
            heads=4
        ).to(device)

        print("\n--- Starting Training Loop Example ---")
        
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            
            print(f"\n--- mini-batch [{batch_idx}] Generated HeteroData Object ---")
            print(batch)
            print(f"\nProcessing Batch {batch_idx} with {batch.num_graphs} graphs.")
            
            #forward pass
            out = model(
                batch.x_dict, 
                batch.edge_index_dict, 
                batch.edge_attr_dict,
                batch.u
            )
            
            print(f"Batch {batch_idx}:")
            print(f"Batch Size: {batch.num_graphs}")
            print(f"Activation Probs: {out['activation']}")
            print(f"Assignment Probs: {out['assignment']}")
            print(f"Sequence Probs: {out['sequence']}")
            
            #example backward pass
            # loss = criterion(out['activation'], batch['operator'].y) ...
            # loss.backward()
            # optimizer.step()
            
            if batch_idx >= 1: break #limit to 2 batches, just for demo




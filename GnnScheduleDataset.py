import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import os
import glob
import re
from GnnDataInstanceBuilder import GnnDataInstanceBuilder
from MultiCriteriaGNNModel import MultiCriteriaGNNModel


#file paths
MISSION_BATCH_DIR = "./datasets/mini-batch/Batch10M_distanced.csv"
UDC_TYPES_DIR = "./datasets/WM_UDC_TYPE.csv"
MISSION_BATCH_TRAVEL_DIR = "./datasets/mini-batch/Batch10M_travel_distanced.csv"
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
                fork_path):
        
        self.schedule_dir = schedule_dir
        self.builder = GnnDataInstanceBuilder()
        self.pallet_types_file_path = pallet_types_file_path
        self.fork_path = fork_path
        
        #discover all schedule files
        #pattern: schedule..._1_A...B...H...0.csv
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
            schedule_file_path=item['schedule']
        )
        
        return data

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #init Dataset
    dataset = GnnScheduleDataset(
        schedule_dir=SCHEDULE_DIR,
        mission_base_path=MISSION_BATCH_DIR,
        edge_base_path=MISSION_BATCH_TRAVEL_DIR,
        pallet_types_file_path=UDC_TYPES_DIR,
        fork_path=FORK_LIFTS_DIR
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




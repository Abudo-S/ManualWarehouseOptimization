import torch
import networkx as nx

class ScheduleValidator:
    def __init__(self, batch):
        self.batch = batch
    
    @torch.no_grad()
    def decode_assignment_one_per_order(self, batch, p_assign):
        """
        Assignment feasibility:
        Decodes the assignment head predictions to select at most one operator for each order.
         
         - batch: the input batch containing edge_index_dict for assignment edges
         - p_assign: predicted probabilities for assignment edges (shape: [num_assign_edges, 1])
        Returns:
         - chosen: boolean mask of shape [num_assign_edges] indicating which edges are selected for
        """

        #edges: operator -> order
        src, dst = batch.edge_index_dict[('operator','assign','order')]
        p = p_assign.view(-1)

        num_orders = batch['order'].num_nodes
        chosen = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)

        #for each order j, pick the edge (i->j) with max probability
        #(if you want allow unassigned, only choose if max_p >= thr)
        max_p = torch.full((num_orders,), -1e9, device=p.device)
        max_e = torch.full((num_orders,), -1, dtype=torch.long, device=p.device)

        for e in range(p.numel()):
            j = dst[e].item()
            if p[e] > max_p[j]:
                max_p[j] = p[e]
                max_e[j] = e

        valid_orders = (max_e >= 0)
        chosen[max_e[valid_orders]] = True
        return chosen  #boolean mask over assignment edges
    
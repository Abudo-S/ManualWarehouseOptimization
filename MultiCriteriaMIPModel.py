from pyomo.environ import *

class MultiCriteriaMIPModel:
    '''
    Mixed-Integer Programming (MIP) model for minimizing makespan and number of operators used
    in a manual warehouse order picking scenario.
    '''
    
    def __init__(self, *args):
        if len(args) > 0:
            self.model_type = 'concrete'
            self.init_concrete_model(*args)
        else:
            self.model_type = 'abstract'
            self.init_abstract_model()
    
    def init_abstract_model(self):
        '''
        Initialize the MIP model with variables, objective function and constraints.
        Values for sets and parameters will be provided later via data files or data portals.
        '''
        model = AbstractModel()

        #MIP Definition
        #Sets
        model.J = Set()         #orders/missions
        model.I_max = Set()     #maximum pool of operators
        model.U = Set()         #orders/missions pallet types
        model.J_prime = Set()   #(missions + virtual base node)

        #Parameters
        model.P = Param(model.I_max, model.J)         #processing Time: P[i, j]
        model.T = Param(model.J_prime, model.J_prime) #travel Time: T[j, k] (distance matrix)
        model.Q = Param(model.I_max, model.U)         #skill score : Q[i, u]
        model.O = Param(model.J, model.U)             #order/mission pallet type : O[j, u]
        model.H_fixed = Param()                       #fixed shift capacity (ex. 480 minutes)
        model.Alpha = Param()                         #makespan weight (for Z)
        model.Beta = Param()                          #Operator count weight (for sum of y_i)
        model.M = Param()                             #big-M constant (ex. 10000)

        self._init_variables_and_constaints(model)
    
    def init_concrete_model(self, 
                 missions,
                 operators, 
                 pallet_types,
                 missions_with_base,
                 processing_times,
                 travel_times,
                 skill_scores,
                 mission_pallet_types,
                 h_fixed=480,
                 alpha=1.0,
                 beta=1.0, #100
                 M=10000
                ):
        '''
        Initialize the MIP model with variables, objective function and constraints.
        Values are provided via constructor arguments for sets and parameters.
        '''
        model = ConcreteModel()

        #MIP Definition
        #Sets
        model.J = Set(initialize=missions)                   #orders/missions
        model.I_max = Set(initialize=operators)              #maximum pool of operators
        model.U = Set(initialize=pallet_types)               #orders/missions pallet types
        model.J_prime = Set(initialize=missions_with_base)   #(missions + virtual base node)

        #Parameters
        #in case of missing data, use big-M as default value (so the corresponding assignment/sequencing will be avoided in optimal solution)
        model.M = Param(initialize=M)                                                               #big-M constant (ex. 10000)
        model.P = Param(model.I_max, model.J, initialize=processing_times, default=model.M)         #processing Time: P[i, j]
        model.T = Param(model.J_prime, model.J_prime, initialize=travel_times, default=model.M)     #travel Time: T[j, k] (distance matrix)
        model.Q = Param(model.I_max, model.U, initialize=skill_scores, default=0)                   #skill score : Q[i, u]
        model.O = Param(model.J, model.U, initialize=mission_pallet_types, default=1)               #order/mission pallet type : O[j, u]
        model.H_fixed = Param(initialize=h_fixed)                                                   #fixed shift capacity (ex. 480 minutes)
        model.Alpha = Param(initialize=alpha)                                                       #makespan weight (for Z)
        model.Beta = Param(initialize=beta)                                                         #Operator count weight (for sum of y_i)

        self._init_variables_and_constaints(model)
        
    def _init_variables_and_constaints(self, model):

        #Binary Variables
        #y[i]: 1 if operator i is activated/used
        model.y = Var(model.I_max, domain=Binary) #operator activation

        #z[i, j, k]: 1 if operator i travels from j to k (sequencing/linking flow)
        model.z = Var(model.I_max, model.J_prime, model.J_prime, domain=Binary) #linking flow

        #x[i, j]: 1 if order j is assigned to operator i
        model.x = Var(model.I_max, model.J, domain=Binary) #order assignment

        #Continuous Variables
        #S[j]: start time of order j
        model.S = Var(model.J, bounds=(0, None))

        #C[j]: completion time of order j
        model.C = Var(model.J, bounds=(0, None))

        #S_first[i]: start time of operator i's first task "departure time from base" (for capacity check)
        model.S_first = Var(model.I_max, bounds=(0, None))

        #C_last[i]: completion time including return to Base (for capacity check)
        model.C_last = Var(model.I_max, bounds=(0, None))

        #Z: overall makespan (max completion time for sequence of missions)
        model.Z = Var(bounds=(0, None))

        #Objective Function
        def objective_rule(model):
            '''
            Multi-criteria objective function: minimize weighted sum of makespan and number of operators used.
            Total Weighted Score = (alpha * makespan) + (beta * total activated operators)
            '''
            return model.Alpha * model.Z + model.Beta * sum(model.y[i] for i in model.I_max)

        #Constraints
        #scheduling and time constraints
        def completion_time_rule(model, j):
            '''
            Completion time: the completion time of order j.
            C[j] = S[j] + sum over i of (x[i,j] * P[i,j])
            '''
            return model.C[j] == model.S[j] + sum(model.x[i, j] * model.P[i, j] for i in model.I_max)

        def sequencing_rule(model, i, j, k):
            '''
            Sequencing/linking flow: Non-Overlapping sequencing.
            S[k] >= C[j] + T[j,k] - M * (1 - z[i,j,k])  for all i in I_max, j in J, k in J, j != k
            '''
            if j != k and j in model.J and k in model.J:
                return model.S[k] >= model.C[j] + model.T[j, k] - model.M * (1 - model.z[i, j, k])
            
            return Constraint.Skip

        #mission assignment and flow constraints
        def assignment_rule(model, j):
            '''
            Mission assignment: each mission j is assigned to exactly one operator i
            '''
            return sum(model.x[i, j] for i in model.I_max) == 1
        
        def operator_skill_rule(model, i, j):
            """
            Mission pallet-type skill: each mission j can only be assigned to operator i if the operator has skill score for the pallet type of the mission.
            If a mission's pallet type is not compatible with any operator'skill score, the mission remains unassigned! leading to unfeasible solution
            [Pre-validation: Each mission's pallet type needs to be in at least one operator score].
            """
            return sum(model.O[j, u] * model.Q[i, u] for u in model.U) >= model.x[i, j]
        
        # def flow_conservation_rule(model, i, k):
        #     '''
        #     Flow conservation: Inflow must equal outflow for every order.
        #     for each operator i and mission/node j,
        #     the sum of flow entering j must equal sum of flow leaving j w.r.t. another mission k,
        #     linking x_ij and z_ijk
        #     '''
        #     inflow = sum(model.z[i, j, k] for j in model.J_prime if j != k)
        #     outflow = sum(model.z[i, k, j] for j in model.J_prime if j != k)

        #     #return inflow == outflow and inflow == model.x[i, k] #error, pyomo tries to evaluate both expressions as one with logical AND
        #     #pyomo rule function that explicitly returns the equality relationship as a tuple
        #     return (inflow == outflow, inflow == model.x[i, k]) #concatenated constraints might create indexing issue with pyomo, so it'd be better to split them
        
        def flow_in_out_rule(model, i, k):
            '''
            Flow conservation: Inflow must equal outflow for every order.
            for each operator i and mission/node j,
            the sum of flow entering j must equal sum of flow leaving j w.r.t. another mission k,
            linking x_ij and z_ijk
            '''
            inflow = sum(model.z[i, j, k] for j in model.J_prime if j != k)
            outflow = sum(model.z[i, k, j] for j in model.J_prime if j != k)

            return inflow == outflow

        def flow_assignment_rule(model, i, k):
            '''
            Flow assignment: the total inflow to mission k for operator i must equal to the assignment variable x[i,k]
            '''
            inflow = sum(model.z[i, j, k] for j in model.J_prime if j != k)

            return inflow == model.x[i, k]
        
        # def base_flow_rule(model, i):
        #     '''
        #     Base flow: Operator leaves and returns to the base if activated.
        #     for each operator i, the flow out of the base node (0) must equal
        #     the flow into the base node (0), which equals y[i]
        #     '''
        #     outflow_from_base = sum(model.z[i, 0, k] for k in model.J)
        #     inflow_to_base = sum(model.z[i, j, 0] for j in model.J)

        #     #pyomo rule function that explicitly returns the equality relationship as a tuple
        #     return (outflow_from_base == model.y[i], inflow_to_base == model.y[i]) #concatenated constraints might create indexing issue with pyomo, so it'd be better to split them

        def base_outflow_rule(model, i):
            '''
            Base flow: Operator leaves and returns to the base if activated.
            for each operator i, the flow out of the base node (0) must equal
            the flow into the base node (0), which equals y[i]
            pt.1. Sum of flow leaving the Base (j=0) to any service order k
            ''' 
            outflow_from_base = sum(model.z[i, 0, k] for k in model.J)

            return outflow_from_base == model.y[i]
        
        def base_inflow_rule(model, i):
            '''            
            Base flow: Operator leaves and returns to the base if activated.
            for each operator i, the flow out of the base node (0) must equal
            the flow into the base node (0), which equals y[i]
            pt.2. Sum of flow entering the Base (k=0) from any service order j
            '''
            inflow_to_base = sum(model.z[i, j, 0] for j in model.J)
            
            return inflow_to_base == model.y[i]

        #resource capacity and makespan constraints
        def c_last_rule(model, i, j):
            '''
            C_last (arrival back at base): it requires a Big-M constraint (enabling) to select the completion time of the last order + return time.
            C_last[i] >= C[j] + T[j,0] - M * (1 - z[i,j,0])
            where [0] is the index of the virtual base node.
            '''
            return model.C_last[i] >= model.C[j] + model.T[j, 0] - (2 * model.M) * (1 - model.z[i, j, 0])

        def capacity_check_rule(model, i):
            '''
            Capacity check: ensure that the total time (based on last order) for each operator i does not exceed fixed shift capacity if activated.
            '''
            return model.C_last[i] <= model.H_fixed * model.y[i]

        def makespan_rule(model, i):
            '''
            Makespan definition: ensure that the makespan Z is at least the completion time including return to base for each operator i.
            '''
            return model.Z >= model.C_last[i]

        #assign objective and constraints to the model
        model.Objective = Objective(rule=objective_rule, sense=minimize)
        model.Sequencing = Constraint(model.I_max, model.J, model.J, rule=sequencing_rule)
        model.CompletionTime = Constraint(model.J, rule=completion_time_rule)

        model.Assignment = Constraint(model.J, rule=assignment_rule)
        model.OperatorSkill = Constraint(model.I_max, model.J, rule=operator_skill_rule)
        #model.FlowConservation = Constraint(model.I_max, model.J, rule=flow_conservation_rule) #splitted into two separate constraints
        model.FlowInflowOutflow = Constraint(model.I_max, model.J, rule=flow_in_out_rule)
        model.FlowAssignment = Constraint(model.I_max, model.J, rule=flow_assignment_rule)

        #model.BaseFlow = Constraint(model.I_max, rule=base_flow_rule) #splitted into two separate constraints
        model.BaseOutflow = Constraint(model.I_max, rule=base_outflow_rule)
        model.BaseInflow = Constraint(model.I_max, rule=base_inflow_rule)
        model.CLastDefinition = Constraint(model.I_max, model.J, rule=c_last_rule)
        model.CapacityCheck = Constraint(model.I_max, rule=capacity_check_rule)
        model.MakespanDefinition = Constraint(model.I_max, rule=makespan_rule)

        self.model = model

    def solve(self, data_file:str=None, data_portal:DataPortal=None, solver_name='glpk'):
        '''
        data_file: path to the data file for the MIP model parameters.
        data_portal: the data portal object that contains all parameters data.
        Solve the MIP model with the provided data_file/data_portal.
        sorver_name: name of the solver to use {glpk, cbc, groubi} (default: 'glpk').
        '''
        
        #SolverFactory("gurobi", solver_io="direct")
        solver = SolverFactory(solver_name) 

        instance = None
        results = None
        if data_file is not None or data_portal is not None:
            assert isinstance(self.model, AbstractModel), "the model must be an AbstractModel to use data files!"
            instance = self.model.create_instance(data_portal) if data_portal is not None else self.model.create_instance(data_file)
            results = solver.solve(instance, tee=True) # 'tee=True' prints the solver log to the console

        else:
            assert isinstance(self.model, ConcreteModel), "the model must be an ConcreteModel to use it directly!"
            results = solver.solve(self.model, tee=True) # 'tee=True' prints the solver log to the console
        
        #it's possible to use dataPortal to load data directly from a dictionary or other sources
        #data = DataPortal(model=self.model)
        #data.load(name='P', data=P_data) #P_data should have the same dimensions as model.P
        #instance = model.create_instance(data)

        return instance, results
    
    def display_solution(self, instance=None):
        '''
        Display the solution of the MIP model.
        '''
        instance.display() if instance is not None else self.model.display()
    
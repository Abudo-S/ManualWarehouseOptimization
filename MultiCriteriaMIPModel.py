from pyomo.environ import *
from pyomo.solvers.plugins.solvers.cplex_persistent import CPLEXPersistent
from pyomo.gdp import Disjunction
import cplex

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
        model.M_Time = Param()                        #big-M constant for time calculations (ex. 10000)

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
                 M_Time=10000,
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
        #model.I_max = Set(initialize=sorted(operators))     #maximum pool of operators
        model.I_max = Set(initialize=operators)              #maximum pool of operators
        model.U = Set(initialize=pallet_types)               #orders/missions pallet types
        model.J_prime = Set(initialize=missions_with_base)   #(missions + virtual base node)

        #Parameters
        #in case of missing data, use big-M as default value (so the corresponding assignment/sequencing will be avoided in optimal solution)
        model.M = Param(initialize=M)                                                               #big-M constant (ex. 10000)
        model.M_Time = Param(initialize=M_Time)                                                     #big-M constant for time calculations (ex. 10000)
        model.P = Param(model.I_max, model.J, initialize=processing_times, default=1)               #processing Time: P[i, j]
        #Note that assigning big-M as default might create infeasible solution if no valid travel time exists for all mission pairs or base-mission pairs
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
        model.S = Var(model.J, bounds=(0, model.H_fixed))

        #C[j]: completion time of order j
        model.C = Var(model.J, bounds=(0, model.H_fixed))

        #S_first[i]: start time of operator i's first task "departure time from base/mission" (for capacity check)
        model.S_first = Var(model.I_max, bounds=(0, model.H_fixed))

        #C_last[i]: completion time including return to Base (for capacity check)
        model.C_last = Var(model.I_max, bounds=(0, model.H_fixed))

        # u[i, j]: MTZ variable, sequence position (rank) of mission j in operator i's route
        model.u = Var(model.I_max, model.J, bounds=(0, len(model.J)))

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
                return model.S[k] >= model.C[j] + model.T[j, k] - model.M_Time * (1 - model.z[i, j, k])
            
            return Constraint.Skip

        def sequencing_disjunction_rule(model, i, j, k):
            if j == k: 
                return Disjunction.Skip
            return [
                [model.z[i,j,k] == 1, model.S[k] >= model.C[j] + model.T[j,k]],
                [model.z[i,j,k] == 0]
            ]

        #mission assignment and flow constraints
        def assignment_rule(model, j):
            '''
            Mission assignment: each mission j is assigned to exactly one operator i
            '''
            return sum(model.x[i, j] for i in model.I_max) == 1
        
        def activation_link_rule(model, i, j):
            '''
            Activation link: an operator i can only be assigned to mission j if the operator is activated.
            '''
            return model.x[i, j] <= model.y[i]
        
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

        def symmetry_break_rule(model, i):
            '''
            Symmetry breaking: to reduce symmetric/interchangable solutions, we need to
            enforce an ordering on operator activation.
            y[i] <= y[i-1] for all i in I_max, i != first operator
            '''

            #i_list = sorted(list(model.I_max))
            i_list = list(model.I_max)
            if i == i_list[0]: 
                return Constraint.Skip
            
            current_index = i_list.index(i)
            previous_operator = i_list[current_index - 1]
            
            #operator i can only be active if operator i-1 is active.
            return model.y[i] <= model.y[previous_operator] 
        
        #MTZ constraints (prevent sub-tours) 
        def mtz_ordering_M_rule(model, i, j, k):
            '''
            MTZ Ordering: eliminates sub-tours not involving the base.
            u[i, k] >= u[i, j] + 1 - M * (1 - z[i, j, k]) for all i in I_max, j in J, k in J, j != k
            '''
            if j != k and j in model.J and k in model.J:
                return model.u[i, k] >= model.u[i, j] + 1 - model.M * (1 - model.z[i, j, k])
            
            return Constraint.Skip

        def mtz_ordering_rule(model, i, j, k):
            """
            MTZ Ordering (alternative form):
            u[i,j] - u[i,k] + |J| * z[i,j,k] <= |J| - 1  for all i, j != k
            """
            if j == k or j not in model.J or k not in model.J:
                return Constraint.Skip

            nJ = len(model.J)
            return model.u[i, j] - model.u[i, k] + nJ * model.z[i, j, k] <= nJ - 1
        
        def mtz_assignment_rule(model, i, j):
            '''
            MTZ Assignment: ensure u[i,j] is 0 if mission j is not assigned to operator i.
            u[i, j] <= |J| * x[i, j] 
            it forces u[i,j] to 0 when x[i,j] = 0.
            '''
            return model.u[i, j] <= len(model.J) * model.x[i, j]

        def mtz_base_start_rule(model, i, j):
            '''
            MTZ Base Start: ensures the rank of the first mission is at least 1.
            u[i, j] >= 1 * z[i, 0, j]
            if z[i, 0, j] = 1 (operator i goes from base to mission j), then u[i, j] >= 1.
            '''
            return model.u[i, j] >= model.z[i, 0, j]
        
        #resource capacity and makespan constraints
        def c_last_rule(model, i, j):
            '''
            C_last (arrival back at base): it requires a Big-M constraint (enabling) to select the completion time of the last order + return time.
            C_last[i] >= C[j] + T[j,0] - M * (1 - z[i,j,0])
            where [0] is the index of the virtual base node.
            '''
            return model.C_last[i] >= model.C[j] + model.T[j, 0] -  model.H_fixed * (1 - model.z[i, j, 0])

        def capacity_check_rule(model, i):
            '''
            Capacity check: ensure that the total time (based on last order) for each operator i does not exceed fixed shift capacity if activated.
            '''
            return model.C_last[i] <= model.H_fixed * model.y[i]
        
        def completion_capacity_check_rule(model, j):
            '''
            Capacity check: ensure that the total time (based on completion time) for each operator i does not exceed fixed shift capacity if activated.
            '''
            return model.C[j] <= model.H_fixed

        def makespan_rule(model, i):
            '''
            Makespan definition: ensure that the makespan Z is at least the completion time including return to base for each operator i.
            '''
            return model.Z >= model.C_last[i] #original definition including return to base
        
        
        def makespan_rule_no_return(model, j):
            '''
            Makespan definition: ensure that the makespan Z is at least the completion time.
            '''
            return model.Z >= model.C[j] #substitutes definition ignoring return to base time

        #assign objective and constraints to the model
        #objective function
        model.Objective = Objective(rule=objective_rule, sense=minimize)

        #scheduling and time constraints
        #model.Sequencing = Constraint(model.I_max, model.J, model.J, rule=sequencing_rule)
        model.indicator_con = Disjunction(model.I_max, model.J, model.J, rule=sequencing_disjunction_rule)
        model.CompletionTime = Constraint(model.J, rule=completion_time_rule)

        #mission assignment and flow constraints
        model.Assignment = Constraint(model.J, rule=assignment_rule)
        model.ActivationLink = Constraint(model.I_max, model.J, rule=activation_link_rule)
        model.OperatorSkill = Constraint(model.I_max, model.J, rule=operator_skill_rule)
        #model.FlowConservation = Constraint(model.I_max, model.J, rule=flow_conservation_rule) #splitted into two separate constraints
        model.FlowInflowOutflow = Constraint(model.I_max, model.J, rule=flow_in_out_rule)
        model.FlowAssignment = Constraint(model.I_max, model.J, rule=flow_assignment_rule)
        #model.BaseFlow = Constraint(model.I_max, rule=base_flow_rule) #splitted into two separate constraints
        model.BaseOutflow = Constraint(model.I_max, rule=base_outflow_rule)
        model.BaseInflow = Constraint(model.I_max, rule=base_inflow_rule)
        model.SymmetryBreak = Constraint(model.I_max, rule=symmetry_break_rule)
        
        # #MTZ constraints
        model.MTZOrdering = Constraint(model.I_max, model.J, model.J, rule=mtz_ordering_rule)
        model.MTZAssignment = Constraint(model.I_max, model.J, rule=mtz_assignment_rule)
        model.MTZBaseStart = Constraint(model.I_max, model.J, rule=mtz_base_start_rule)

        #resource capacity and makespan constraints
        #model.CLastDefinition = Constraint(model.I_max, model.J, rule=c_last_rule)
        #model.CapacityCheck = Constraint(model.I_max, rule=capacity_check_rule)
        model.CompletionCapacityCheck = Constraint(model.J, rule=completion_capacity_check_rule) #substitutes CapacityCheck & CLastDefinition but ignores return to base time
        #model.MakespanDefinition = Constraint(model.I_max, rule=makespan_rule)
        model.MakespanDefinition = Constraint(model.J, rule=makespan_rule_no_return) #substitutes MakespanDefinition but ignores return to base time

        self.model = model
        TransformationFactory('gdp.bigm').apply_to(self.model) #apply the big-M reformulation for disjunctions

    def solve(self, 
              data_file:str=None, 
              data_portal:DataPortal=None, 
              time_limit=None, 
              mip_gap=None, 
              solver_name='glpk'):
        '''
        data_file: path to the data file for the MIP model parameters.
        data_portal: the data portal object that contains all parameters data.
        Solve the MIP model with the provided data_file/data_portal.
        sorver_name: name of the solver to use {glpk, cbc, groubi, cplex} (default: 'glpk').
        time_limit: time limit for the glpk solver in seconds (default: None).
        mip_gap: MIP gap tolerance percent for the solver (default: None) [ex. 0.05 #stop if within 5% of optimal].
        '''

        #SolverFactory("gurobi", solver_io="direct")
        solver = SolverFactory(solver_name) 
        
        # if not solver.available():
        #     print("Solver still not found. Check your PATH and Python environment.")
            
        #add solver options based on the selected solver
        if solver_name == "glpk":
            if time_limit is not None:
                solver.options['tmlim'] = time_limit  
            if mip_gap is not None: 
                solver.options['mipgap'] = mip_gap 
            # solver.options['tolpiv'] = 1e-7      # Bounding tolerance
            # solver.options['tolint'] = 1e-7      # Integer tolerance (must be small)
            # solver.options['scale'] = 'on'       # Force internal scaling

        elif solver_name == "cbc":
            if time_limit is not None:
                solver.options['sec'] = time_limit  
            if mip_gap is not None:
                solver.options['ratio'] = mip_gap
        elif solver_name == "gurobi":
            solver.options['NumericFocus'] = 'yes'
            if time_limit is not None:
                solver.options['TimeLimit'] = time_limit  
            if mip_gap is not None:
                solver.options['MIPGap'] = mip_gap
        elif solver_name == "cplex":
            #solver.options['emphasis numerical'] = 3
            if time_limit is not None:
                solver.options['timelimit'] = time_limit  
            if mip_gap is not None:
                solver.options['mipgap'] = mip_gap
        elif solver_name == "cplex_persistent":
            #solver.options['emphasis numerical'] = 3
            if time_limit is not None:
                solver.options['timelimit'] = time_limit  
            if mip_gap is not None:
                solver.options['mipgap'] = mip_gap

        instance = None
        results = None
        if data_file is not None or data_portal is not None:
            assert isinstance(self.model, AbstractModel), "the model must be an AbstractModel to use data files!"
            instance = self.model.create_instance(data_portal) if data_portal is not None else self.model.create_instance(data_file)
            results = solver.solve(instance, tee=True) # 'tee=True' prints the solver log to the console

        elif solver_name != "cplex_persistent":
            assert isinstance(self.model, ConcreteModel), "the model must be an ConcreteModel to use it directly!"
            results = solver.solve(self.model, tee=True) # 'tee=True' prints the solver log to the console
        else:
            #for cplex_persistent, the instance is already set in the solver
            solver.set_instance(self.model)
            results = solver.solve(tee=True)
        
        #it's possible to use dataPortal to load data directly from a dictionary or other sources
        #data = DataPortal(model=self.model)
        #data.load(name='P', data=P_data) #P_data should have the same dimensions as model.P
        #instance = model.create_instance(data)

        # if results.solver.termination_condition == TerminationCondition.optimal:
        #     self.model.load(results)
        #     # Proceed with post-processing (e.g., printing objective, variables)
        #     print("Optimal solution found.")
        # else:
        #     # Print the specific condition instead of crashing
        #     print(f"Solver terminated with non-optimal condition: {results.solver.termination_condition}")
        #     print(f"Solver status: {results.solver.status}")

        if results.solver.termination_condition == TerminationCondition.infeasible:
            print("Infeasible! Launching Conflict Refiner...")
            
            # Access the raw CPLEX engine object
            cpx_engine = solver._solver_model
            
            # Trigger the conflict refinement
            cpx_engine.conflict.refine()
            
            # Option A: Save the conflict to a file (Easiest to read)
            cpx_engine.conflict.write("conflict_report.clp")
            print("Conflict details saved to 'conflict_report.clp'")
            
            # Option B: Print conflict status directly
            # Note: This returns a list of statuses for all constraints/bounds
            conflicts = cpx_engine.conflict.get()
            for i, status in enumerate(conflicts):
                if status == cpx_engine.conflict.group_status.member:
                    print(f"Index {i} is a member of the minimal conflict.")

        return instance, results
    
    def display_solution(self, instance=None):
        '''
        Display the solution of the MIP model.
        '''
        instance.display() if instance is not None else self.model.display()
    
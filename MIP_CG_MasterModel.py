from pyomo.environ import *

class MIP_CG_MasterModel:
    '''
    Column Generation Master Problem (MP) for the Multi-Criteria VRP.
    Formulated as a Set Partitioning Problem to select pre-calculated routes.
    The model assumes that a Set of Routes (R) has already been generated (initially, this set might be empty or contain a few simple routes). 
    The variables λr are initially defined as continuous for the Linear Program (LP) relaxation, which is solved in the Column Generation loop.
    '''
    def __init__(self, routes_data, initial_params):
        #routes_data:
        # {
        #   'route_1': {'cost': C1, 'time': T1, 'operator': i1, 'visits': {j1: 1, j2: 1, ...}},
        #   'route_2': {...},
        #   ...
        # }
        self.routes_data = routes_data
        self.I_max = initial_params['operators']  #set of available operators
        self.J = initial_params['missions']       #set of missions/orders
        self.Alpha = initial_params['alpha']
        self.Beta = initial_params['beta']

        self.model = ConcreteModel()
        self._init_sets()
        self._init_variables()
        self._init_objective()
        self._init_constraints()

    def _init_sets(self):
        #set of all currently generated routes
        self.model.R = Set(initialize=self.routes_data.keys())
        self.model.J = Set(initialize=self.J)
        self.model.I_max = Set(initialize=self.I_max)

        #cost and Assignment parameters for each route
        self.model.RouteCost = Param(self.model.R, initialize={r: d['cost'] for r, d in self.routes_data.items()})
        self.model.RouteTime = Param(self.model.R, initialize={r: d['time'] for r, d in self.routes_data.items()})
        self.model.RouteOperator = Param(self.model.R, initialize={r: d['operator'] for r, d in self.routes_data.items()})

        #a_jr: 1 if mission j is covered by route r
        a_jr_init = {}
        for r, data in self.routes_data.items():
            for j in self.model.J:
                a_jr_init[(j, r)] = data['visits'].get(j, 0)
        
        self.model.a_jr = Param(self.model.J, self.model.R, initialize=a_jr_init)

    def _init_variables(self):
        #lambda[r]: 1 if route r is selected (Continuous for LP relaxation)
        self.model.Lambda = Var(self.model.R, domain=NonNegativeReals) 

        #y[i]: 1 if operator i is activated (Binary/Continuous, depending on decomposition stage)
        self.model.y = Var(self.model.I_max, domain=NonNegativeReals, bounds=(0, 1))

        self.model.Z = Var(bounds=(0, None))

    def _init_objective(self):
        #The objective is the combined weighted cost of the selected routes (makespan) 
        #plus the weighted cost of the activated operators.
        
        #NOTE: A simplified approach is to fold the makespan constraint into the route cost,
        #but the common way to handle makespan (Min-Max) is via the Z variable:
        
        #Simplified route cost = (Cost of the route)
        #However, to explicitly include makespan in the objective function of the master problem,
        #we need to minimize the weighted sum:
        
        def objective_rule(model):
            #the total cost of the selected routes (Cost_r is usually just distance/time, 
            #but we use Z for the Min-Max Makespan part.)
            #total Weighted Score = (alpha * makespan Z) + (beta * total activated operators)
            return self.Alpha * model.Z + self.Beta * sum(model.y[i] for i in model.I_max)

        self.model.Objective = Objective(rule=objective_rule, sense=minimize)

    def _init_constraints(self):
        #set Partitioning/Covering Constraint: Every mission must be covered exactly once.
        def cover_rule(model, j):
            '''
            Sum of Lambda for all routes r that visit mission j must equal 1.
            '''
            return sum(model.Lambda[r] * model.a_jr[j, r] for r in model.R) == 1
        
        #these dual variables (shadow prices) will be passed to the Subproblem
        self.model.MissionCoverage = Constraint(self.model.J, rule=cover_rule)

        #operator Activation Constraint: If any route r belonging to operator i is selected (Lambda > 0), 
        #then operator i must be activated (y_i must be 1 or Lambda is bounded by y_i).
        def operator_use_rule(model, i):
            '''
            The sum of Lambda for all routes r operated by i cannot exceed operator i's activation (y_i).
            Since Lambda are continuous and y_i is continuous (0-1), this enforces that 
            y_i is at least the fraction of work assigned to operator i.
            '''
            #filter routes for a specific operator i
            routes_for_i = [r for r in model.R if model.RouteOperator[r] == i]
            
            if not routes_for_i:
                return Constraint.Skip 
            
            return sum(model.Lambda[r] for r in routes_for_i) <= model.y[i] * len(model.J) # Multiplying by |J| ensures sum of Lambda <= |J|

        self.model.OperatorActivation = Constraint(self.model.I_max, rule=operator_use_rule)

        #Makespan Constraint: Z must be greater than or equal to the time of the longest selected route.
        #it's the tricky part in the LP relaxation. Z should ideally be the max of all selected routes.
        #Since Z is in the objective, we need Z to be >= each selected route's time.
        #In the LP relaxation, we enforce Z >= the TIME of the selected solution set.

        def makespan_def_rule(model):
            '''
            A linear formulation of the max-time (makespan) is usually achieved by 
            setting Z >= C_r for ALL possible routes r. Here, we must include the route time in the objective 
            or use Z >= C_r * Lambda_r.
            Using the aggregated approach (since Lambda is fractional in LP):
            Z must be greater than or equal to the *weighted average* or *total* time. 
            The simplest approach is Z >= C_r. But this is not correct for Min-Max.
            The correct formulation for Min-Max (Makespan) in a set partitioning master problem is:
            Z >= Time_r * Lambda_r for all r. This forces Z to respect the time of the selected routes.
            '''
            #enforce that the makespan Z is greater than or equal to the time of every selected route.
            return model.Z >= sum(model.RouteTime[r] * model.Lambda[r] for r in model.R)
            
        self.model.MakespanDefinition = Constraint(rule=makespan_def_rule)

    def get_dual_values(self):
        '''
        Extracts the dual variables (shadow prices) from the MissionCoverage constraint
        to be passed to the Subproblem.
        '''
        if not hasattr(self.model.MissionCoverage, 'dual'):
            #the solver must support dual extraction (e.g., Gurobi, CPLEX, GLPK)
            print("Dual values not available. Ensure the solver supports duals.")
            return {}
            
        return {j: self.model.dual[self.model.MissionCoverage[j]] for j in self.model.J}
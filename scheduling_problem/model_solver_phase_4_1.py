import pyomo.environ as pyo
import pandas as pd
import numpy as np

def create_and_solve_model(
    jobs,                    # Set of jobs (j)
    machines,                # Set of machines (i)
    time_periods,            # Set of time periods (t)
    e_i,                     # Dictionary of costs e_i for each machine i
    f_i,                     # Dictionary of setup costs f_i for each machine i
    M,                       # Fixed cost parameter
    d_i,                     # Dictionary of job durations d_i for each machine i
    n_i,                     # Dictionary of minimum job requirements n_i for each machine i
    m_t,                     # Dictionary of machine capacity limits for time t
    NB,                      # Budget constraint parameter
    machine_dependencies,    # List of tuples (k, k+1) representing machine dependencies
    shared_resources,        # List of lists where each sublist is a set of machines M' sharing resources
    unavailable_machines,    # Dictionary of (machine, time) tuples where the machine is unavailable
    c_i,                     # Dictionary of cooldown periods c_i for each machine i
):
    # Create the model
    model = pyo.ConcreteModel()
    
    # Sets
    model.I = pyo.Set(initialize=machines)        # Machines
    model.J = pyo.Set(initialize=jobs)            # Jobs
    model.T = pyo.Set(initialize=time_periods)    # Time periods
    
    # Define variables
    model.x = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if machine i processes job j at time t
    model.y = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)  # 1 if setup occurs for machine i, job j at time t
    model.p = pyo.Var(model.T, domain=pyo.Binary)                    # 1 if payment occurs at time t
    model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)          # Savings at time t
    
    # Constraint 1: Total cost at each time period
    def cost_constraint_rule(model, t):
        return sum(e_i[i] * model.x[i, t, j] + f_i[i] * model.y[i, t, j] 
                   for i in model.I for j in model.J) - M * model.p[t] - model.s[t] <= 0
    model.cost_constraint = pyo.Constraint(model.T, rule=cost_constraint_rule)
    
    # Constraint 2: Savings calculation
    def savings_rule(model, t):
        if t == time_periods[0]:
            return model.s[t] == 0
        else:
            prior_times = [t_prime for t_prime in time_periods if t_prime < t]
            return model.s[t] == sum(
                M * model.p[t_prime] - sum(e_i[i] * model.x[i, t_prime, j] + f_i[i] * model.y[i, t_prime, j] 
                                           for i in model.I for j in model.J)
                for t_prime in prior_times
            )
    model.savings_calculation = pyo.Constraint(model.T, rule=savings_rule)
    
    # Constraint 3: Budget constraint
    def budget_constraint_rule(model, t):
        return model.s[t] <= NB
    model.budget_constraint = pyo.Constraint(model.T, rule=budget_constraint_rule)
    
    # Constraint 4: Minimum job requirements
    def min_job_requirement_rule(model, i):
        return sum(model.x[i, t, j] for t in model.T for j in model.J) >= n_i[i] * d_i[i]
    model.min_job_requirement = pyo.Constraint(model.I, rule=min_job_requirement_rule)
    
    # Constraint 5: Machine can handle at most one job at a time
    def machine_one_job_rule(model, i, t):
        return sum(model.x[i, t, j] for j in model.J) <= 1
    model.machine_one_job = pyo.Constraint(model.I, model.T, rule=machine_one_job_rule)
    
    # Constraint 6: At most one setup per machine per time period
    def max_one_setup_rule(model, i, t):
        return sum(model.y[i, t, j] for j in model.J) <= 1
    model.max_one_setup = pyo.Constraint(model.I, model.T, rule=max_one_setup_rule)
    
    # Constraint 7: Machine capacity constraints
    def machine_capacity_rule(model, t):
        return sum(e_i[i] * model.x[i, t, j] for i in model.I for j in model.J) <= m_t[t]
    model.machine_capacity = pyo.Constraint(model.T, rule=machine_capacity_rule)
    
    # Constraint 8: Unavailable machines
    for i, t in unavailable_machines:
        def unavailable_machine_rule(model, i=i, t=t):
            if i is None: # TODO: see if this is right
                i = 1
            return sum(model.x[i, t, j] for j in model.J) == 0
        model.add_component(f'unavailable_machine_{i}_{t}', pyo.Constraint(rule=unavailable_machine_rule))
    
    # Constraint 9: Shared resources constraints
    for idx, machines_subset in enumerate(shared_resources):
        def shared_resource_rule(model, t, machines_subset=machines_subset):
            return sum(model.x[i, t, j] for i in machines_subset for j in model.J) <= 1
        model.add_component(f'shared_resource_{idx}', pyo.Constraint(model.T, rule=shared_resource_rule))
    
    # Constraint 10: Setup and continuity relationship
    def setup_continuity_rule(model, i, t, j):
        if t == time_periods[0]:
            return model.y[i, t, j] <= model.x[i, t, j]
        else:
            prev_t = time_periods[time_periods.index(t) - 1]
            return pyo.inequality(model.y[i, t, j], model.x[i, t, j], model.y[i, t, j] + model.x[i, prev_t, j])
    model.setup_continuity = pyo.Constraint(model.I, model.T, model.J, rule=setup_continuity_rule)
    
    # Constraint 11: Dependencies between machines
    for k, k_plus_1 in machine_dependencies:
        def dependency_rule(model, t, j, k=k, k_plus_1=k_plus_1):
            # Check if we're at the first time period
            if t == time_periods[0]:
                return model.y[k_plus_1, t, j] <= 0  # Cannot start dependent job at first time period
            else:
                prior_times = [t_prime for t_prime in time_periods if t_prime < t]
                if d_i[k] > 0:  # Avoid division by zero
                    return model.y[k_plus_1, t, j] <= sum(model.x[k, t_prime, j] for t_prime in prior_times) / d_i[k]
                else:
                    return pyo.Constraint.Skip
        model.add_component(f'dependency_{k}_{k_plus_1}', pyo.Constraint(model.T, model.J, rule=dependency_rule))
    
    # Constraint 12: Cooldown period constraints
    def cooldown_rule(model, i, t, j):
        # If we're at the beginning or don't have enough periods for cooldown, skip
        if t in time_periods[:c_i[i]]:
            return pyo.Constraint.Skip
        else:
            # Get the relevant time periods for cooldown
            cooldown_periods = [t_prime for t_prime in time_periods 
                                if t - c_i[i] <= t_prime < t]
            if cooldown_periods:
                return model.y[i, t, j] <= sum(1 - sum(model.x[i, t_prime, j2] for j2 in model.J) 
                                             for t_prime in cooldown_periods) / c_i[i]
            else:
                return pyo.Constraint.Skip
    model.cooldown = pyo.Constraint(model.I, model.T, model.J, rule=cooldown_rule)
    
    # Constraint 13: Job completion constraint
    def job_completion_rule(model, i, j):
        return sum(model.x[i, t, j] for t in model.T) == d_i[i]
    model.job_completion = pyo.Constraint(model.I, model.J, rule=job_completion_rule)
    
    # Objective: Minimize total cost
    def objective_rule(model):
        return sum(e_i[i] * model.x[i, t, j] + f_i[i] * model.y[i, t, j] 
                  for i in model.I for t in model.T for j in model.J) - sum(M * model.p[t] for t in model.T)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    # Solve the model using GLPK
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=True)
    
    # Check if the model solved successfully
    if (results.solver.status == pyo.SolverStatus.ok and 
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("Optimal solution found!")
        
        # Extract solution
        solution = {
            'x': {(i, t, j): model.x[i, t, j].value for i in model.I for t in model.T for j in model.J if model.x[i, t, j].value > 0.5},
            'y': {(i, t, j): model.y[i, t, j].value for i in model.I for t in model.T for j in model.J if model.y[i, t, j].value > 0.5},
            'p': {t: model.p[t].value for t in model.T if model.p[t].value > 0.5},
            's': {t: model.s[t].value for t in model.T},
            'objective_value': pyo.value(model.objective)
        }
        
        return model, solution
    else:
        print("No optimal solution found. Solver status:", results.solver.status)
        print("Termination condition:", results.solver.termination_condition)
        return model, None

# Example usage
if __name__ == "__main__":
    # Define example data
    jobs = [1, 2, 3]
    machines = [1,2,3]
    time_periods = list(range(1, 11))  # 10 time periods
    
    # Machine costs
    e_i = {1: 10, 2: 15, 3: 20}
    
    # Setup costs
    f_i = {1: 5, 2: 8, 3: 12}
    
    # Fixed cost parameter
    M = 100
    
    # Job durations for each machine
    d_i = {1: 2, 2: 3, 3: 2}
    
    # Minimum job requirements
    n_i = {1: 1, 2: 1, 3: 1}
    
    # Machine capacity limits for each time period
    m_t = {t: 50 for t in time_periods}
    
    # Budget constraint
    NB = 200
    
    # Machine dependencies (k, k+1)
    machine_dependencies = [(1, 2), (2, 3)]
    
    # Shared resources
    shared_resources = [[1, 2], [2, 3]]
    
    # Unavailable machines at specific times
    unavailable_machines = [(1, 3), (2, 5)]
    
    # Cooldown periods
    c_i = {1: 1, 2: 2, 3: 1}
    
    # Create and solve the model
    model, solution = create_and_solve_model(
        jobs=jobs,
        machines=machines,
        time_periods=time_periods,
        e_i=e_i,
        f_i=f_i,
        M=M,
        d_i=d_i,
        n_i=n_i,
        m_t=m_t,
        NB=NB,
        machine_dependencies=machine_dependencies,
        shared_resources=shared_resources,
        unavailable_machines=unavailable_machines,
        c_i=c_i
    )
    
    # Display solution if found
    if solution:
        print("\nMachine schedules (x variables):")
        for (i, t, j), val in solution['x'].items():
            print(f"Machine {i} processes job {j} at time {t}")
        
        print("\nSetup operations (y variables):")
        for (i, t, j), val in solution['y'].items():
            print(f"Setup for machine {i}, job {j} at time {t}")
        
        print("\nPayment times (p variables):")
        for t, val in solution['p'].items():
            print(f"Payment made at time {t}")
        
        print("\nSavings (s variables):")
        for t, val in solution['s'].items():
            print(f"Savings at time {t}: {val}")
        
        print("\nTotal objective value:", solution['objective_value'])

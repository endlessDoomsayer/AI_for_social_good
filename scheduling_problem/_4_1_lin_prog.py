from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import combine_data

# THIS USES SCIP AND NOT CSP

def find_min(M, N, data):
    
    print(f"Searching M={M}, N={N}")

    I = data["I"]
    J = data["J"]
    T = data["T"]
    n_jobs = data["n_jobs"]
    d = data["d"]
    e = data["e"]
    f = data["f"]
    c_b = data["c_b"]
    c_p = data["c_p"]
    c = data["c"]
    p = data["p"]
    mmm = data["mmm"]
    silent_periods = data["silent_periods"]
    M_shared = data["M_shared"]
    M_dependencies = data["M_dependencies"]
    B = data["B"]
    T_MAX = data["T_MAX"]
    THRESHOLD_FOR_JOB_J_AND_I = data["THRESHOLD_FOR_JOB_J_AND_I"]
    MACHINES = data["MACHINES"]

    # Create the linear solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print('No suitable solver found.')
            exit()

    # Variables - Create for ALL i,j,t combinations first
    x = {}
    y = {}
    for i in I:
        for j in J:
            for t in T:
                x[i, t, j] = solver.BoolVar(f'x_{i}_{t}_{j}')
                y[i, t, j] = solver.BoolVar(f'y_{i}_{t}_{j}')

    # Storage variables
    s = {t: solver.NumVar(0, solver.infinity(), f's_{t}') for t in T}


    # Fix variables for jobs that a machine can't do (matching Pyomo exactly)
    for i in I:
        for j in J:
            if j > n_jobs[i]:
                for t in T:
                    x[i, t, j].SetBounds(0, 0)  # Fix to 0
                    y[i, t, j].SetBounds(0, 0)  # Fix to 0


    # Constraint 1: Energy balance constraint
    # Pyomo: sum(e[i] * m.x[i, t, j] + f[i] * m.y[i, t, j] for i in m.I for j in m.J) - M * p[t] - m.s[t] <= 0
    # Rearranged: sum(e[i] * x[i,t,j] + f[i] * y[i,t,j]) <= M * p[t] + s[t]
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), M * p[t])
        constraint.SetCoefficient(s[t], -1) 
        for i in I:
            for j in J:
                constraint.SetCoefficient(x[i, t, j], e[i])
                constraint.SetCoefficient(y[i, t, j], f[i])

    # Constraint 2: Storage computation constraint
    # Pyomo: m.s[t] <= sum(M * p[tp] - sum(e[i] * m.x[i, tp, j] + f[i] * m.y[i, tp, j] for i in m.I for j in m.J) for tp in range(1, t))
    for t in T:
        if t == 1:
            solver.Add(s[t] == 0)  # Starting with empty storage
        else:
            # s[t] <= sum_{tp=1}^{t-1}[M * p[tp] - sum_{i,j}(e[i] * x[i,tp,j] + f[i] * y[i,tp,j])]
            # Rearranged: s[t] + sum_{tp=1}^{t-1} sum_{i,j}(e[i] * x[i,tp,j] + f[i] * y[i,tp,j]) <= sum_{tp=1}^{t-1} M * p[tp]
            rhs_value = sum(M * p[tp] for tp in range(1, t))
            constraint = solver.Constraint(-solver.infinity(), rhs_value)
            constraint.SetCoefficient(s[t], 1)
            
            for tp in range(1, t):
                for i in I:
                    for j in J:
                        constraint.SetCoefficient(x[i, tp, j], e[i])
                        constraint.SetCoefficient(y[i, tp, j], f[i])

    # Constraint 3: Battery capacity constraint
    # Pyomo: m.s[t] <= N * B
    for t in T:
        solver.Add(s[t] <= N * B)

    # Constraint 4: Each job must run for required duration
    # Pyomo: sum(model.x[i, t, j] for t in T) == d[i] (for j <= n_jobs[i])
    for i in I:
        for j in J:
            if j <= n_jobs[i]:  # Only add constraint for required jobs
                constraint = solver.Constraint(d[i], d[i])
                for t in T:
                    constraint.SetCoefficient(x[i, t, j], 1)

    # Constraint 5: Each machine can do one job at a time
    # Pyomo: sum(m.x[i, t, j] for j in m.J) <= 1
    for i in I:
        for t in T:
            constraint = solver.Constraint(-solver.infinity(), 1)
            for j in J:
                constraint.SetCoefficient(x[i, t, j], 1)

    # Constraint 6: Same for starting
    # Pyomo: sum(m.y[i, t, j] for j in m.J) <= 1
    for i in I:
        for t in T:
            constraint = solver.Constraint(-solver.infinity(), 1)
            for j in J:
                constraint.SetCoefficient(y[i, t, j], 1)

    # Constraint 7: Max energy constraint
    # Pyomo: sum(e[i] * m.x[i, t, j] for i in m.I for j in m.J) <= mmm[t]
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), mmm[t])
        for i in I:
            for j in J:
                constraint.SetCoefficient(x[i, t, j], e[i])

    # Constraint 8: Silent periods for machines
    # Pyomo: sum(model.x[i, t, j] for j in J) == 0
    for i in silent_periods:
        if i in I:  # Make sure machine i is in our set
            for t in silent_periods[i]:
                if t in T and t <= T_MAX:
                    constraint = solver.Constraint(0, 0)
                    for j in J:
                        constraint.SetCoefficient(x[i, t, j], 1)

    # Constraint 9: Shared resource constraint
    # Pyomo: sum(model.x[i, t, j] for i in machines_in_group for j in J) <= 1
    for t in T:
        for group in M_shared:
            machines_in_group = [i for i in group if i in I]
            if machines_in_group:
                constraint = solver.Constraint(-solver.infinity(), 1)
                for i in machines_in_group:
                    for j in J:
                        constraint.SetCoefficient(x[i, t, j], 1)

    # Constraint 10: Start implies run and ensure continuity constraint
    # Pyomo: For t==1: m.x[i, t, j] == m.y[i, t, j]
    #        For t>1:  m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t-1, j]
    for i in I:
        for t in T:
            for j in J:
                if t == 1:
                    # x[i,1,j] == y[i,1,j]
                    solver.Add(x[i, t, j] == y[i, t, j])
                else:
                    # x[i,t,j] <= y[i,t,j] + x[i,t-1,j]
                    # Rearranged: x[i,t,j] - y[i,t,j] - x[i,t-1,j] <= 0
                    constraint = solver.Constraint(-solver.infinity(), 0)
                    constraint.SetCoefficient(x[i, t, j], 1)
                    constraint.SetCoefficient(y[i, t, j], -1)
                    constraint.SetCoefficient(x[i, t-1, j], -1)

    # Constraint 10b: When you start a job, you must run it
    # Pyomo: m.y[i, t, j] <= m.x[i, t, j]
    for i in I:
        for t in T:
            for j in J:
                solver.Add(y[i, t, j] <= x[i, t, j])

    # Constraint 11: Dependency constraint
    # Pyomo: For t==1: model.y[kp1, t, j] == 0
    #        For t>1:  model.y[kp1, t, j] <= prev_completions / d[k]
    for (k, kp1) in M_dependencies:
        if k in I and kp1 in I:
            for t in T:
                for j in J:
                    if j <= n_jobs.get(k, 0) and j <= n_jobs.get(kp1, 0):  # Both machines must be able to do job j
                        if t == 1:
                            solver.Add(y[kp1, t, j] == 0)
                        else:
                            # y[kp1,t,j] <= sum(x[k,tp,j] for tp in range(1,t)) / d[k]
                            # Rearranged: y[kp1,t,j] * d[k] <= sum(x[k,tp,j] for tp in range(1,t))
                            constraint = solver.Constraint(-solver.infinity(), 0)
                            constraint.SetCoefficient(y[kp1, t, j], d[k])
                            for tp in range(1, t):
                                constraint.SetCoefficient(x[k, tp, j], -1)

    # Constraint 12: Cooldown constraint (simplified version from Pyomo)
    # Pyomo: y[i,t,j] <= cooldown_sum / c[i]
    # Where cooldown_sum = sum(1 - sum(x[i,tp,jj] for jj in J) for tp in range(t-c[i], t))
    for i in I:
        for t in T:
            for j in J:
                if t > c[i]:
                    # Simplified: machine must be idle for at least 1 period before starting
                    if t > 1:
                        # y[i,t,j] + sum(x[i,t-1,jj] for jj in J) <= 1
                        constraint = solver.Constraint(-solver.infinity(), 1)
                        constraint.SetCoefficient(y[i, t, j], 1)
                        for jj in J:
                            constraint.SetCoefficient(x[i, t-1, jj], 1)

    # Constraint 13: Job must be completed before threshold
    # Pyomo: model.x[i, t, j].fix(0) for t > THRESHOLD_FOR_JOB_J_AND_I[(i, j)]
    for i in I:
        for j in J:
            if j <= n_jobs[i]:  # Only apply for required jobs
                threshold = THRESHOLD_FOR_JOB_J_AND_I.get((i, j), T_MAX)
                for t in range(threshold + 1, T_MAX + 1):
                    if t in T:
                        x[i, t, j].SetBounds(0, 0)  # Fix to 0

    # Set solver parameters
    solver.SetTimeLimit(300000)  # 5 minutes timeout

    # Solve
    status = solver.Solve()

    status_dict = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE", 
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        return True
    else:
        return False

def print_solution(M,N,data,filename):
    I = data["I"]
    J = data["J"]
    T = data["T"]
    n_jobs = data["n_jobs"]
    d = data["d"]
    e = data["e"]
    f = data["f"]
    c_b = data["c_b"]
    c_p = data["c_p"]
    c = data["c"]
    p = data["p"]
    mmm = data["mmm"]
    silent_periods = data["silent_periods"]
    M_shared = data["M_shared"]
    M_dependencies = data["M_dependencies"]
    B = data["B"]
    T_MAX = data["T_MAX"]
    THRESHOLD_FOR_JOB_J_AND_I = data["THRESHOLD_FOR_JOB_J_AND_I"]
    MACHINES = data["MACHINES"]

    # Create the linear solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print('No suitable solver found.')
            exit()

    # Variables - Create for ALL i,j,t combinations first
    x = {}
    y = {}
    for i in I:
        for j in J:
            for t in T:
                x[i, t, j] = solver.BoolVar(f'x_{i}_{t}_{j}')
                y[i, t, j] = solver.BoolVar(f'y_{i}_{t}_{j}')

    # Storage variables
    s = {t: solver.NumVar(0, solver.infinity(), f's_{t}') for t in T}


    # Fix variables for jobs that a machine can't do (matching Pyomo exactly)
    for i in I:
        for j in J:
            if j > n_jobs[i]:
                for t in T:
                    x[i, t, j].SetBounds(0, 0)  # Fix to 0
                    y[i, t, j].SetBounds(0, 0)  # Fix to 0


    # Constraint 1: Energy balance constraint
    # Pyomo: sum(e[i] * m.x[i, t, j] + f[i] * m.y[i, t, j] for i in m.I for j in m.J) - M * p[t] - m.s[t] <= 0
    # Rearranged: sum(e[i] * x[i,t,j] + f[i] * y[i,t,j]) <= M * p[t] + s[t]
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), M * p[t])
        constraint.SetCoefficient(s[t], -1) 
        for i in I:
            for j in J:
                constraint.SetCoefficient(x[i, t, j], e[i])
                constraint.SetCoefficient(y[i, t, j], f[i])

        # 2. Storage computation constraint
        for t in T:  # TODO: this has changed, change in all the other models
            if t == 1:
                solver.Add(s[t] == 0)  # Assume starting with empty storage
            else:
                # s[t] = s[t-1] + production[t-1] - consumption[t-1]
                constraint = solver.Constraint(M*p[t], M*p[t])
                constraint.SetCoefficient(s[t], 1)
                constraint.SetCoefficient(s[t - 1], -1)

                # Add consumption from previous period
                for i in I:
                    for j in J:
                        constraint.SetCoefficient(x[i, t - 1, j], e[i])  # Running consumption
                        constraint.SetCoefficient(y[i, t - 1, j], f[i])  # Startup consumption

    # Constraint 3: Battery capacity constraint
    # Pyomo: m.s[t] <= N * B
    for t in T:
        solver.Add(s[t] <= N * B)

    # Constraint 4: Each job must run for required duration
    # Pyomo: sum(model.x[i, t, j] for t in T) == d[i] (for j <= n_jobs[i])
    for i in I:
        for j in J:
            if j <= n_jobs[i]:  # Only add constraint for required jobs
                constraint = solver.Constraint(d[i], d[i])
                for t in T:
                    constraint.SetCoefficient(x[i, t, j], 1)

    # Constraint 5: Each machine can do one job at a time
    # Pyomo: sum(m.x[i, t, j] for j in m.J) <= 1
    for i in I:
        for t in T:
            constraint = solver.Constraint(-solver.infinity(), 1)
            for j in J:
                constraint.SetCoefficient(x[i, t, j], 1)

    # Constraint 6: Same for starting
    # Pyomo: sum(m.y[i, t, j] for j in m.J) <= 1
    for i in I:
        for t in T:
            constraint = solver.Constraint(-solver.infinity(), 1)
            for j in J:
                constraint.SetCoefficient(y[i, t, j], 1)

    # Constraint 7: Max energy constraint
    # Pyomo: sum(e[i] * m.x[i, t, j] for i in m.I for j in m.J) <= mmm[t]
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), mmm[t])
        for i in I:
            for j in J:
                constraint.SetCoefficient(x[i, t, j], e[i])

    # Constraint 8: Silent periods for machines
    # Pyomo: sum(model.x[i, t, j] for j in J) == 0
    for i in silent_periods:
        if i in I:  # Make sure machine i is in our set
            for t in silent_periods[i]:
                if t in T and t <= T_MAX:
                    constraint = solver.Constraint(0, 0)
                    for j in J:
                        constraint.SetCoefficient(x[i, t, j], 1)

    # Constraint 9: Shared resource constraint
    # Pyomo: sum(model.x[i, t, j] for i in machines_in_group for j in J) <= 1
    for t in T:
        for group in M_shared:
            machines_in_group = [i for i in group if i in I]
            if machines_in_group:
                constraint = solver.Constraint(-solver.infinity(), 1)
                for i in machines_in_group:
                    for j in J:
                        constraint.SetCoefficient(x[i, t, j], 1)

    # Constraint 10: Start implies run and ensure continuity constraint
    # Pyomo: For t==1: m.x[i, t, j] == m.y[i, t, j]
    #        For t>1:  m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t-1, j]
    for i in I:
        for t in T:
            for j in J:
                if t == 1:
                    # x[i,1,j] == y[i,1,j]
                    solver.Add(x[i, t, j] == y[i, t, j])
                else:
                    # x[i,t,j] <= y[i,t,j] + x[i,t-1,j]
                    # Rearranged: x[i,t,j] - y[i,t,j] - x[i,t-1,j] <= 0
                    constraint = solver.Constraint(-solver.infinity(), 0)
                    constraint.SetCoefficient(x[i, t, j], 1)
                    constraint.SetCoefficient(y[i, t, j], -1)
                    constraint.SetCoefficient(x[i, t-1, j], -1)

    # Constraint 10b: When you start a job, you must run it
    # Pyomo: m.y[i, t, j] <= m.x[i, t, j]
    for i in I:
        for t in T:
            for j in J:
                solver.Add(y[i, t, j] <= x[i, t, j])

    # Constraint 11: Dependency constraint
    # Pyomo: For t==1: model.y[kp1, t, j] == 0
    #        For t>1:  model.y[kp1, t, j] <= prev_completions / d[k]
    for (k, kp1) in M_dependencies:
        if k in I and kp1 in I:
            for t in T:
                for j in J:
                    if j <= n_jobs.get(k, 0) and j <= n_jobs.get(kp1, 0):  # Both machines must be able to do job j
                        if t == 1:
                            solver.Add(y[kp1, t, j] == 0)
                        else:
                            # y[kp1,t,j] <= sum(x[k,tp,j] for tp in range(1,t)) / d[k]
                            # Rearranged: y[kp1,t,j] * d[k] <= sum(x[k,tp,j] for tp in range(1,t))
                            constraint = solver.Constraint(-solver.infinity(), 0)
                            constraint.SetCoefficient(y[kp1, t, j], d[k])
                            for tp in range(1, t):
                                constraint.SetCoefficient(x[k, tp, j], -1)

    # Constraint 12: Cooldown constraint (simplified version from Pyomo)
    # Pyomo: y[i,t,j] <= cooldown_sum / c[i]
    # Where cooldown_sum = sum(1 - sum(x[i,tp,jj] for jj in J) for tp in range(t-c[i], t))
    for i in I:
        for t in T:
            for j in J:
                if t > c[i]:
                    # Simplified: machine must be idle for at least 1 period before starting
                    if t > 1:
                        # y[i,t,j] + sum(x[i,t-1,jj] for jj in J) <= 1
                        constraint = solver.Constraint(-solver.infinity(), 1)
                        constraint.SetCoefficient(y[i, t, j], 1)
                        for jj in J:
                            constraint.SetCoefficient(x[i, t-1, jj], 1)

    # Constraint 13: Job must be completed before threshold
    # Pyomo: model.x[i, t, j].fix(0) for t > THRESHOLD_FOR_JOB_J_AND_I[(i, j)]
    for i in I:
        for j in J:
            if j <= n_jobs[i]:  # Only apply for required jobs
                threshold = THRESHOLD_FOR_JOB_J_AND_I.get((i, j), T_MAX)
                for t in range(threshold + 1, T_MAX + 1):
                    if t in T:
                        x[i, t, j].SetBounds(0, 0)  # Fix to 0

    # Set solver parameters
    solver.SetTimeLimit(300000)  # 5 minutes timeout

    # Solve
    status = solver.Solve()

    status_dict = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE", 
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }
    print(f"Status: {status_dict.get(status, 'UNKNOWN')}")
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Solution found!")
        
        storage_values = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        print("\nMachine Schedules:")
        for i in I:
            print(f"\nMachine {i} Schedule:")
            for j in J:
                if j <= n_jobs[i]:  # Only check feasible jobs
                    start_times = [t for t in T if y[i, t, j].solution_value() > 0.5]
                    operative_times = [t for t in T if x[i, t, j].solution_value() > 0.5]
                    if start_times:
                        print(f"  Job {j} starts at t={start_times}")
                    if operative_times:
                        print(f"  Job {j} operates at t={operative_times}")
                        for t in operative_times:
                            ax1.bar(t, 1, bottom=i - 1, color=f'C{j % 10}', edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Machine')
        ax1.set_yticks(range(0, MACHINES))
        ax1.set_yticklabels([f'Machine {i}' for i in I])
        ax1.set_title('Machine Schedule')
        # Create legend for jobs that actually exist
        legend_jobs = set()
        for i in I:
            for j in J:
                if j <= n_jobs[i]:
                    legend_jobs.add(j)
        #ax1.legend([f'Job {j}' for j in sorted(legend_jobs)], loc='upper right')

        print("\nStorage Levels:")
        for t in T:
            storage_val = s[t].solution_value()
            storage_values.append(storage_val)
            #if t <= 10:  # Show first 10 time periods for brevity
            print(f"  t={t}: {storage_val:.2f}")

        ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
        battery_capacity = N * B
        ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Energy Storage')
        ax2.set_title('Energy Storage Levels')
        ax2.legend()

        plt.tight_layout()
        plt.savefig("schedule_visualization_"+filename+".png", format="png", dpi=300, bbox_inches='tight')
        print("\nSchedule visualization saved as 'schedule_visualization_"+filename+".svg")
        plt.show()

        # Print solution statistics
        print(f"\nSolution Statistics:")
        print(f"Total variables: {solver.NumVariables()}")
        print(f"Total constraints: {solver.NumConstraints()}")
        print(f"Solve time: {solver.wall_time()} ms")

    else:
        print("Failed to find an optimal solution.")
        print("Possible issues:")
        print("1. Problem is infeasible")
        print("2. Solver timeout") 
        print("3. Numerical issues with constraints")
        
        print(f"\nSolver info:")
        print(f"Total variables: {solver.NumVariables()}")
        print(f"Total constraints: {solver.NumConstraints()}")
        
def solve(M, N):
    data = combine_data.get_data()
    #find_min(M,N,data)
    print_solution(M,N,data,"4_1_lin_prog")

if __name__ == "__main__":
    solve(4912,45)
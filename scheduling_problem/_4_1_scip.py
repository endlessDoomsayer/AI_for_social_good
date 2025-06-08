from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import combine_data

def add_constraints(solver, data, M, N, x, y, s):
    I = data["I"]
    J = data["J"]
    T = data["T"]
    n_jobs = data["n_jobs"]
    d = data["d"]
    e = data["e"]
    f = data["f"]
    c = data["c"]
    p = data["p"]
    mmm = data["mmm"]
    silent_periods = data["silent_periods"]
    M_shared = data["M_shared"]
    M_dependencies = data["M_dependencies"]
    B = data["B"]
    T_MAX = data["T_MAX"]
    THRESHOLD_FOR_JOB_J_AND_I = data["THRESHOLD_FOR_JOB_J_AND_I"]

    # Fix variables for jobs that a machine can't do (matching Pyomo exactly)
    for i in I:
        for j in J:
            if j > n_jobs[i]:
                for t in T:
                    x[i, t, j].SetBounds(0, 0)  # Fix to 0
                    y[i, t, j].SetBounds(0, 0)  # Fix to 0

    # 1. Energy balance constraint
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), M*p[t])
        constraint.SetCoefficient(s[t], -1)
        for i in I:
            for j in J:
                constraint.SetCoefficient(x[i, t, j], e[i])
                constraint.SetCoefficient(y[i, t, j], f[i])

    # 2. Storage computation constraint
    for t in T:
        if t == 1:
            solver.Add(s[t] == 0)
        else:
            constraint = solver.Constraint(M * p[t-1], M * p[t-1])
            constraint.SetCoefficient(s[t], 1)
            constraint.SetCoefficient(s[t - 1], -1)

            for i in I:
                for j in J:
                    constraint.SetCoefficient(x[i, t - 1, j], e[i])
                    constraint.SetCoefficient(y[i, t - 1, j], f[i])

    # 3. Battery capacity constraint
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), N*B)
        constraint.SetCoefficient(s[t], 1)

    # 4. Each job must run for required duration
    for i in I:
        for j in J:
            if j <= n_jobs[i]:
                constraint = solver.Constraint(d[i], d[i])
                for t in T:
                    constraint.SetCoefficient(x[i, t, j], 1)

    # 5. Each machine can do one job at a time
    for i in I:
        for t in T:
            constraint = solver.Constraint(-solver.infinity(), 1)
            for j in J:
                constraint.SetCoefficient(x[i, t, j], 1)

    # 6. Same for starting
    for i in I:
        for t in T:
            constraint = solver.Constraint(-solver.infinity(), 1)
            for j in J:
                constraint.SetCoefficient(y[i, t, j], 1)

    # 7. Max energy constraint
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), mmm[t])
        for i in I:
            for j in J:
                constraint.SetCoefficient(x[i, t, j], e[i])

    # 8. Silent periods for machines (some machines must be off at specific times)
    for i in silent_periods:
        if i in I:
            for t in silent_periods[i]:
                if t in T and t <= T_MAX:
                    for j in J:
                        constraint = solver.Constraint(0, 0)
                        constraint.SetCoefficient(x[i, t, j], 1)

    # 9. Shared resource constraint
    for t in T:
        for group in M_shared:
            machines_in_group = [i for i in group if i in I]
            if machines_in_group:
                constraint = solver.Constraint(-solver.infinity(), 1)
                for i in machines_in_group:
                    for j in J:
                        constraint.SetCoefficient(x[i, t, j], 1)

    # 10. Start implies run and ensure continuity constraint
    for i in I:
        for t in T:
            for j in J:
                if t == 1:
                    # For the first time period: x[i,t,j] = y[i,t,j]
                    solver.Add(x[i, t, j] == y[i, t, j])
                else:
                    # For subsequent periods: x[i,t,j] <= y[i,t,j] + x[i,t-1,j]
                    constraint = solver.Constraint(-solver.infinity(), 0)
                    constraint.SetCoefficient(x[i, t, j], 1)
                    constraint.SetCoefficient(y[i, t, j], -1)
                    constraint.SetCoefficient(x[i, t - 1, j], -1)

    # 10b. When you start a job, you must run it
    for i in I:
        for t in T:
            for j in J:
                solver.Add(y[i, t, j] <= x[i, t, j])

    # 11. Dependency constraint
    for (k, kp1) in M_dependencies:
        if k in I and kp1 in I:
            for t in T:
                for j in J:
                    if j <= n_jobs.get(k, 0) and j <= n_jobs.get(kp1, 0):  # Both machines must be able to do job j
                        if t == 1:
                            solver.Add(y[kp1, t, j] == 0)  # Cannot start dependent job at first time period
                        else:
                            # Job on machine kp1 can start only if job on machine k was completed
                            constraint = solver.Constraint(-solver.infinity(), 0)
                            constraint.SetCoefficient(y[kp1, t, j], d[k])
                            for tp in range(1, t):
                                constraint.SetCoefficient(x[k, tp, j], -1)

    # 12. Cooldown constraint.
    for i in I:
        for t in T:
            if t > c[i]:
                constraint = solver.Constraint(-solver.infinity(), n_jobs[i] * c[i])
                for j in J:
                    constraint.SetCoefficient(y[i, t, j], c[i])
                    for tp in range(t - c[i], t):
                        constraint.SetCoefficient(x[i, tp, j], 1)

    # 13. Job must be completed before threshold
    for i in I:
        for j in J:
            if j <= n_jobs[i]:
                threshold = THRESHOLD_FOR_JOB_J_AND_I.get((i, j), T_MAX)
                for t in range(threshold + 1, T_MAX + 1):
                    if t in T:
                        x[i, t, j].SetBounds(0, 0)  # Fix to 0

def find_min(M, N, data):
    
    print(f"Searching M={M}, N={N}")

    # Create the linear solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print('No suitable solver found.')
            exit()

    T = data["T"]
    I = data["I"]
    J = data["J"]

    # Variables
    x = {}
    y = {}
    for i in I:
        for j in J:
            for t in T:
                x[i, t, j] = solver.BoolVar(f'x_{i}_{t}_{j}')
                y[i, t, j] = solver.BoolVar(f'y_{i}_{t}_{j}')

    # Storage variables
    s = {t: solver.NumVar(0, solver.infinity(), f's_{t}') for t in T}

    add_constraints(solver, data, M, N, x, y, s)

    # Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        return True
    else:
        return False

def print_solution(M,N,data,filename):
    # Create the linear solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print('No suitable solver found.')
            exit()

    T = data["T"]
    I = data["I"]
    J = data["J"]

    # Variables
    x = {}
    y = {}
    for i in I:
        for j in J:
            for t in T:
                x[i, t, j] = solver.BoolVar(f'x_{i}_{t}_{j}')
                y[i, t, j] = solver.BoolVar(f'y_{i}_{t}_{j}')

    # Storage variables
    s = {t: solver.NumVar(0, solver.infinity(), f's_{t}') for t in T}

    add_constraints(solver, data, M, N, x, y, s)

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
        for i in data["I"]:
            print(f"\nMachine {i} Schedule:")
            n_jobs = data["n_jobs"]
            for j in data["J"]:
                if j <= n_jobs[i]:  # Only check feasible jobs
                    start_times = [t for t in T if y[i, t, j].solution_value() > 0.5]
                    operative_times = [t for t in T if x[i, t, j].solution_value() > 0.5]
                    if start_times:
                        print(f"  Job {j} starts at t={start_times}")
                    if operative_times:
                        print(f"  Job {j} operates at t={operative_times}")
                        for t in operative_times:
                            ax1.bar(t, 1, bottom=i - 1, color=f'C{j % 10}', edgecolor='black', linewidth=0.5)

        MACHINES = data["MACHINES"]

        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Machine')
        ax1.set_yticks(range(0, MACHINES))
        ax1.set_yticklabels([f'Machine {i}' for i in I])
        ax1.set_title('Machine Schedule')

        print("\nStorage Levels:")
        for t in data["T"]:
            storage_val = s[t].solution_value()
            storage_values.append(storage_val)
            print(f"  t={t}: {storage_val:.2f}")

        ax2.plot(data["T"], storage_values, marker='o', linestyle='-', markersize=4)
        battery_capacity = N * data["B"]
        ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Energy Storage')
        ax2.set_title('Energy Storage Levels')
        ax2.legend()

        plt.tight_layout()
        plt.savefig("output/schedule_visualization_"+filename+".png", format="png", dpi=300, bbox_inches='tight')
        print("\nSchedule visualization saved as 'output/schedule_visualization_"+filename+".png")

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
        
def solve(M, N, data = combine_data.get_data()):
    print_solution(M,N,data,"4_1_scip")

if __name__ == "__main__":
    solve(2491,141)
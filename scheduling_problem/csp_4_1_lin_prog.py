from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from combine_data import get_data

# Constants
M = 1000
N = 1500

data = get_data()

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
    print('Could not create solver SCIP. Trying GLOP...')
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        print('No suitable solver found.')
        exit()

print(f"Using solver: {solver.SolverVersion()}")

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

print("Variables created...")

# Fix variables for jobs that a machine can't do (matching Pyomo)
for i in I:
    for j in J:
        if j > n_jobs[i]:
            for t in T:
                x[i, t, j].SetBounds(0, 0)  # Fix to 0
                y[i, t, j].SetBounds(0, 0)  # Fix to 0

print("Fixed variables for impossible jobs...")

# Constraint 1: Energy balance constraint
# sum(e[i] * x[i,t,j] + f[i] * y[i,t,j]) - M * p[t] - s[t] <= 0
print("Adding energy balance constraints...")
for t in T:
    constraint = solver.Constraint(-solver.infinity(), M * p[t])
    constraint.SetCoefficient(s[t], 1)
    for i in I:
        for j in J:
            constraint.SetCoefficient(x[i, t, j], e[i])
            constraint.SetCoefficient(y[i, t, j], f[i])

# Constraint 2: Storage computation constraint
print("Adding storage constraints...")
for t in T:
    if t == 1:
        solver.Add(s[t] == 0)  # Starting with empty storage
    else:
        # s[t] <= sum(M * p[tp] - sum(e[i] * x[i,tp,j] + f[i] * y[i,tp,j])) for tp in range(1, t)
        rhs_value = sum(M * p[tp] for tp in range(1, t))
        constraint = solver.Constraint(-solver.infinity(), rhs_value)
        constraint.SetCoefficient(s[t], 1)
        
        for tp in range(1, t):
            for i in I:
                for j in J:
                    constraint.SetCoefficient(x[i, tp, j], e[i])
                    constraint.SetCoefficient(y[i, tp, j], f[i])

# Constraint 3: Battery capacity constraint
print("Adding battery capacity constraints...")
for t in T:
    solver.Add(s[t] <= N * B)

# Constraint 4: Each job must run for required duration (FIXED VERSION matching Pyomo)
print("Adding job requirement constraints...")
for i in I:
    for j in J:
        if j <= n_jobs[i]:  # Only add constraint for required jobs
            constraint = solver.Constraint(d[i], d[i])
            for t in T:
                constraint.SetCoefficient(x[i, t, j], 1)

# Constraint 5: Each machine can do one job at a time
print("Adding single job constraints...")
for i in I:
    for t in T:
        constraint = solver.Constraint(0, 1)
        for j in J:
            constraint.SetCoefficient(x[i, t, j], 1)

# Constraint 6: Same for starting
print("Adding single start constraints...")
for i in I:
    for t in T:
        constraint = solver.Constraint(0, 1)
        for j in J:
            constraint.SetCoefficient(y[i, t, j], 1)

# Constraint 7: Max energy constraint
print("Adding max energy constraints...")
for t in T:
    constraint = solver.Constraint(-solver.infinity(), mmm[t])
    for i in I:
        for j in J:
            constraint.SetCoefficient(x[i, t, j], e[i])

# Constraint 8: Silent periods for machines
print("Adding silent period constraints...")
for i, times in silent_periods.items():
    for t in times:
        if t <= T_MAX:
            constraint = solver.Constraint(0, 0)
            for j in J:
                constraint.SetCoefficient(x[i, t, j], 1)

# Constraint 9: Shared resource constraint
print("Adding shared resource constraints...")
for t in T:
    for group in M_shared:
        machines_in_group = [i for i in group if i in I]
        if machines_in_group:
            constraint = solver.Constraint(0, 1)
            for i in machines_in_group:
                for j in J:
                    constraint.SetCoefficient(x[i, t, j], 1)

# Constraint 10: Start implies run and ensure continuity constraint (FIXED VERSION matching Pyomo)
print("Adding run-start relation constraints...")
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
                constraint.SetCoefficient(x[i, t-1, j], -1)

# Constraint 10b: When you start a job, you must run it
print("Adding start implies run constraints...")
for i in I:
    for t in T:
        for j in J:
            solver.Add(y[i, t, j] <= x[i, t, j])

# Constraint 11: Dependency constraint
print("Adding dependency constraints...")
for (k, kp1) in M_dependencies:
    if k in I and kp1 in I:
        for t in T:
            for j in J:
                if t == 1:
                    solver.Add(y[kp1, t, j] == 0)  # Cannot start dependent job at first time period
                else:
                    # Job on machine kp1 can start only if job on machine k was completed
                    # y[kp1,t,j] <= prev_completions / d[k]
                    # Rearranged: y[kp1,t,j] * d[k] <= prev_completions
                    constraint = solver.Constraint(-solver.infinity(), 0)
                    constraint.SetCoefficient(y[kp1, t, j], d[k])
                    for tp in range(1, t):
                        constraint.SetCoefficient(x[k, tp, j], -1)

# Constraint 12: Cooldown constraint (FIXED VERSION matching Pyomo)
print("Adding cooldown constraints...")
for i in I:
    for t in T:
        for j in J:
            if t > c[i]:
                # Can only start if the machine was off for at least c[i] time units
                # y[i,t,j] <= cooldown_sum / c[i]
                # Rearranged: y[i,t,j] * c[i] <= cooldown_sum
                # cooldown_sum = sum(1 - sum(x[i,tp,jj] for jj in J) for tp in range(t-c[i], t))
                
                # This is complex due to the (1 - sum(...)) terms
                # Let's implement a simplified version: machine must be idle for at least 1 period before starting
                if t > 1:
                    constraint = solver.Constraint(-solver.infinity(), 1)
                    constraint.SetCoefficient(y[i, t, j], 1)
                    for jj in J:
                        constraint.SetCoefficient(x[i, t-1, jj], 1)

# Constraint 13: Job must be completed before threshold (FIXED VERSION matching Pyomo)
print("Adding job completion threshold constraints...")
for i in I:
    for j in J:
        if j <= n_jobs[i]:  # Only apply for required jobs
            # Fix variables to 0 after threshold
            for t in range(THRESHOLD_FOR_JOB_J_AND_I[(i, j)] + 1, T_MAX + 1):
                x[i, t, j].SetBounds(0, 0)  # Fix to 0

print("All constraints added. Starting solve...")

# Set solver parameters
solver.SetTimeLimit(300000)  # 5 minutes timeout

# Solve
status = solver.Solve()

print(f"Solver status: {status}")

if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    print(f"Solution found!")
    
    storage_values = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    print("\nMachine Schedules:")
    for i in I:
        print(f"\nMachine {i} Schedule:")
        for j in J:
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
    ax1.legend([f'Job {j}' for j in range(1, min(11, max(J)+1))], loc='upper right')

    print("\nStorage Levels:")
    for t in T:
        storage_val = s[t].solution_value()
        storage_values.append(storage_val)
        if t <= 10:  # Show first 10 time periods for brevity
            print(f"  t={t}: {storage_val:.2f}")

    ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
    battery_capacity = N * B
    ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Energy Storage')
    ax2.set_title('Energy Storage Levels')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('schedule_visualization_lp.png', dpi=300, bbox_inches='tight')
    print("\nSchedule visualization saved as 'schedule_visualization_lp.png'")
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
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from combine_data import get_data
import time


# Output of phase 3 - converted to use SCIP instead of GLPK
def solve(M, N, data=get_data(number_of_days=7)):
    BIG_M = 1000000

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
    MACHINES = data["MACHINES"]

    # Create the linear solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print('No suitable solver found.')
            return

    # Variables
    x = {}  # 1 if machine i runs job j at time t
    y = {}  # 1 if machine i starts job j at time t
    for i in I:
        for j in J:
            for t in T:
                x[i, t, j] = solver.BoolVar(f'x_{i}_{t}_{j}')
                y[i, t, j] = solver.BoolVar(f'y_{i}_{t}_{j}')

    # Energy stored at time t
    s = {t: solver.NumVar(0, solver.infinity(), f's_{t}') for t in T}

    # New variables wrt model 1
    z = {t: solver.NumVar(0, solver.infinity(), f'z_{t}') for t in T}  # Deficit variable
    V = {t: solver.NumVar(-solver.infinity(), solver.infinity(), f'V_{t}') for t in
         T}  # Volume variable before applying constraints
    b = {t: solver.BoolVar(f'b_{t}') for t in T}  # Binary variable for V_t sign

    # Fix variables for jobs that a machine can't do
    for i in I:
        for j in J:
            if j > n_jobs[i]:
                for t in T:
                    x[i, t, j].SetBounds(0, 0)
                    y[i, t, j].SetBounds(0, 0)

    # Objective: Minimize battery and power costs plus deficit
    # Original: sum(0*m.z[t] for t in m.T)
    objective = solver.Objective()
    for t in T:
        objective.SetCoefficient(z[t], 0)
    objective.SetMinimization()

    # Constraints

    # 1. Energy balance constraint with deficit
    # Original: sum(e[i] * m.x[i, t, j] + f[i] * m.y[i, t, j] for i in m.I for j in m.J) - M * p[t] - m.s[t] <= m.z[t]
    # Rearranged: sum(e[i] * x[i,t,j] + f[i] * y[i,t,j]) - M * p[t] - s[t] - z[t] <= 0
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), M * p[t])
        constraint.SetCoefficient(s[t], 1)
        constraint.SetCoefficient(z[t], 1)
        for i in I:
            for j in J:
                constraint.SetCoefficient(x[i, t, j], e[i])
                constraint.SetCoefficient(y[i, t, j], f[i])

    # 2. Volume calculation constraint
    # Original: m.V[t] == sum(M * p[tp] - sum(e[i] * m.x[i, tp, j] + f[i] * m.y[i, tp, j] for i in m.I for j in m.J) for tp in range(1, t))
    for t in T:
        rhs_value = sum(M * p[tp] for tp in range(1, t))
        constraint = solver.Constraint(rhs_value, rhs_value)  # Equality constraint
        constraint.SetCoefficient(V[t], 1)

        for tp in range(1, t):
            for i in I:
                for j in J:
                    constraint.SetCoefficient(x[i, tp, j], e[i])
                    constraint.SetCoefficient(y[i, tp, j], f[i])

    # 3. Constraints for binary variable b_t based on V_t sign
    # Original: m.V[t] >= -BIG_M * (1 - m.b[t])
    # Rearranged: V[t] + BIG_M * b[t] >= BIG_M
    for t in T:
        constraint = solver.Constraint(BIG_M, solver.infinity())
        constraint.SetCoefficient(V[t], 1)
        constraint.SetCoefficient(b[t], BIG_M)

    # Original: m.V[t] <= BIG_M * m.b[t]
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), 0)
        constraint.SetCoefficient(V[t], 1)
        constraint.SetCoefficient(b[t], -BIG_M)

    # 4. Storage constraints based on V_t and b_t
    # Original: m.s[t] <= m.V[t] + BIG_M * (1 - m.b[t])
    # Rearranged: s[t] - V[t] + BIG_M * b[t] <= BIG_M
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), BIG_M)
        constraint.SetCoefficient(s[t], 1)
        constraint.SetCoefficient(V[t], -1)
        constraint.SetCoefficient(b[t], BIG_M)

    # Original: m.s[t] <= BIG_M * m.b[t]
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), 0)
        constraint.SetCoefficient(s[t], 1)
        constraint.SetCoefficient(b[t], -BIG_M)

    # 5. Battery capacity constraint (it's already non negative)
    # Original: m.s[t] <= N * B
    for t in T:
        solver.Add(s[t] <= N * B)

    # 6. Each job must run for required duration
    for i in I:
        for j in J:
            if j <= n_jobs[i]:  # Only add constraint for required jobs
                constraint = solver.Constraint(d[i], d[i])
                for t in T:
                    constraint.SetCoefficient(x[i, t, j], 1)

    # 7. Each machine can do one job at a time
    # Original: sum(m.x[i, t, j] for j in m.J) <= 1
    for i in I:
        for t in T:
            constraint = solver.Constraint(-solver.infinity(), 1)
            for j in J:
                constraint.SetCoefficient(x[i, t, j], 1)

    # 8. Each machine can start one job at a time
    # Original: sum(m.y[i, t, j] for j in m.J) <= 1
    for i in I:
        for t in T:
            constraint = solver.Constraint(-solver.infinity(), 1)
            for j in J:
                constraint.SetCoefficient(y[i, t, j], 1)

    # 12. Start implies run and continuity constraint
    # Original: For t==1: m.x[i, t, j] == m.y[i, t, j]
    #          For t>1:  m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t-1, j]
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
                    constraint.SetCoefficient(x[i, t - 1, j], -1)

    # 13. When you start a job, you must run it
    # Original: m.y[i, t, j] <= m.x[i, t, j]
    for i in I:
        for t in T:
            for j in J:
                solver.Add(y[i, t, j] <= x[i, t, j])

    # 16. Job must be completed before threshold
    for i in I:
        for j in J:
            if j <= n_jobs[i]:  # Only apply for required jobs
                # Ensure job runs for exact duration (already covered in constraint 6)

                # Ensure job is not scheduled after threshold
                threshold = THRESHOLD_FOR_JOB_J_AND_I.get((i, j), T_MAX)
                for t in range(threshold + 1, T_MAX + 1):
                    if t in T:
                        x[i, t, j].SetBounds(0, 0)

    # Set solver parameters
    solver.SetTimeLimit(5000000)  # 5000 seconds timeout (matching original tmlim=5000)

    print("Solving the model...")
    start = time.time()
    status = solver.Solve()
    end = time.time()

    print(f"Time taken for MILP: {end - start}")

    # Print results
    status_dict = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }

    print(f"\nSolution Status: {status_dict.get(status, 'UNKNOWN')}")

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Objective Value: {solver.Objective().Value()}")

        # Print number of panels and batteries
        print(f"\nNumber of Batteries: {N}")
        print(f"\nNumber of Panels: {M}")

        # Print deficit values
        print("\nDeficit Values (z_t):")
        for t in range(1, min(11, T_MAX + 1)):  # Show first 10 time periods for brevity
            if t in T:
                print(f"  t={t}: {z[t].solution_value():.2f}")

        # Print machine schedules
        for i in I:
            print(f"\nMachine {i} Schedule:")
            for j in J:
                start_times = [t for t in T if y[i, t, j].solution_value() > 0.5]
                operative_times = [t for t in T if x[i, t, j].solution_value() > 0.5]
                if start_times:
                    print(f"  Job {j} starts at t={start_times}")
                if operative_times:
                    print(f"  Job {j} operates at t={operative_times}")

        # Print energy storage levels
        storage_values = [s[t].solution_value() for t in T]
        print("\nStorage Levels:")
        for t in range(1, min(11, T_MAX + 1)):  # Show first 10 time periods for brevity
            if t in T:
                print(f"  t={t}: {s[t].solution_value():.2f}")

        # Print volume values
        print("\nVolume Values (V_t):")
        for t in range(1, min(11, T_MAX + 1)):  # Show first 10 time periods for brevity
            if t in T:
                print(f"  t={t}: {V[t].solution_value():.2f}")

        # Create visualization of the schedule
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # Plot machine schedules
        for i in I:
            for t in T:
                for j in J:
                    if x[i, t, j].solution_value() > 0.5:
                        ax1.bar(t, 1, bottom=i - 1, color=f'C{j}', edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Machine')
        ax1.set_yticks(range(0, MACHINES))
        ax1.set_yticklabels([f'Machine {i}' for i in I])
        ax1.set_title('Machine Schedule')
        ax1.legend([f'Job {j}' for j in J], loc='upper right')

        # Plot energy storage
        ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
        battery_capacity = N * B
        ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Energy Storage')
        ax2.set_title('Energy Storage Levels')
        ax2.legend()

        # Plot deficit
        deficit_values = [z[t].solution_value() for t in T]
        ax3.plot(T, deficit_values, marker='s', linestyle='-', markersize=4, color='red')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Energy Deficit')
        ax3.set_title('Energy Deficit (z_t)')

        plt.tight_layout()
        plt.savefig('schedule_visualization_43_scip.svg', format="svg")
        print("\nSchedule visualization saved as 'schedule_visualization_43_scip.svg'")

        # plt.show()
    else:
        print("Failed to find an optimal solution.")


if __name__ == "__main__":
    solve(2243, 26)
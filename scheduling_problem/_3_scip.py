from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import time
from combine_data import get_data


def solve(max_time=5000, number_of_days=1, tot_number_of_days=4011):
    # Get data
    data = get_data()

    # Round to 3 decimal places
    def float_to_round(float_list):
        return {x: round(float_list[x], 3) for x in float_list}

    I = data["I"]
    J = data["J"]
    T = data["T"]
    n_jobs = data["n_jobs"]
    d = data["d"]
    e = float_to_round(data["e"])
    f = float_to_round(data["f"])
    c_b = data["c_b"]
    c_p = data["c_p"]
    c_e = data["c_e"]
    c_e *= tot_number_of_days/number_of_days
    c = data["c"]
    p = float_to_round(data["p"])
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
            return None

    # Variables
    M_var = solver.IntVar(0, solver.infinity(), 'M')  # Number of power units
    N_var = solver.IntVar(0, solver.infinity(), 'N')  # Number of batteries

    # Binary variables for machine operations
    x = {}  # 1 if machine i runs job j at time t
    y = {}  # 1 if machine i starts job j at time t
    for i in I:
        for j in J:
            for t in T:
                x[i, t, j] = solver.BoolVar(f'x_{i}_{t}_{j}')
                y[i, t, j] = solver.BoolVar(f'y_{i}_{t}_{j}')

    # Continuous variables
    s = {t: solver.NumVar(0, solver.infinity(), f's_{t}') for t in T}  # Energy stored at time t
    z = {t: solver.NumVar(0, solver.infinity(), f'z_{t}') for t in T}  # Deficit variable

    # Fix variables for jobs that a machine can't do
    for i in I:
        for j in J:
            if j > n_jobs[i]:
                for t in T:
                    x[i, t, j].SetBounds(0, 0)
                    y[i, t, j].SetBounds(0, 0)

    # Objective: Minimize battery and power costs plus deficit
    objective = solver.Objective()
    objective.SetCoefficient(N_var, c_b)
    objective.SetCoefficient(M_var, c_p)
    for t in T:
        objective.SetCoefficient(z[t], c_e)
    objective.SetMinimization()

    # Constraints

    # 1. Energy balance constraint
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), 0)
        constraint.SetCoefficient(M_var, -p[t])
        constraint.SetCoefficient(s[t], -1)
        constraint.SetCoefficient(z[t], -1)
        for i in I:
            for j in J:
                constraint.SetCoefficient(x[i, t, j], e[i])
                constraint.SetCoefficient(y[i, t, j], f[i])

    # 2. Storage computation constraint
    for t in T:
        if t == 1:
            solver.Add(s[t] == 0)  # Assume starting with empty storage
        else:
            # s[t] = s[t-1] + production[t-1] - consumption[t-1]
            constraint = solver.Constraint(0, 0)
            constraint.SetCoefficient(s[t], 1)
            constraint.SetCoefficient(s[t - 1], -1)
            constraint.SetCoefficient(M_var, -p[t-1])
            constraint.SetCoefficient(z[t-1], -1)

            # Add consumption from previous period
            for i in I:
                for j in J:
                    constraint.SetCoefficient(x[i, t - 1, j], e[i])  # Running consumption
                    constraint.SetCoefficient(y[i, t - 1, j], f[i])  # Startup consumption

    # 3. Battery capacity constraint
    for t in T:
        constraint = solver.Constraint(-solver.infinity(), 0)
        constraint.SetCoefficient(s[t], 1)
        constraint.SetCoefficient(N_var, -B)

        # constraint = solver.Constraint(-solver.infinity(), 0)
        # constraint.SetCoefficient(s[t], -1)

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
        if i in I:  # Make sure machine i is in our set
            for t in silent_periods[i]:
                if t in T and t <= T_MAX:
                    for j in J:
                        constraint = solver.Constraint(0, 0)
                        constraint.SetCoefficient(x[i, t, j], 1)

    # 9. Shared resource constraint
    for t in T:
        for group in M_shared:
            machines_in_group = [i for i in group if i in I]  # Ensure the machines are in our defined set
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
        if k in I and kp1 in I:  # Ensure both machines are in our defined set
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
            if j <= n_jobs[i]:  # Only apply for required jobs
                threshold = THRESHOLD_FOR_JOB_J_AND_I.get((i, j), T_MAX)
                for t in range(threshold + 1, T_MAX + 1):
                    if t in T:
                        x[i, t, j].SetBounds(0, 0)  # Fix to 0

    # Solve the model
    solver.SetTimeLimit(1000*max_time)

    print("Solving the model...")
    start = time.time()
    status = solver.Solve()
    end = time.time()

    print(f"Time taken for SCIP on model 3: {end - start:.2f} seconds")

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
        M_value = int(M_var.solution_value())
        N_value = int(N_var.solution_value())
        print(f"\nNumber of Batteries: {N_value}")
        print(f"\nNumber of Panels: {M_value}")

        # Print deficit values
        print("\nDeficit Values (z_t):")
        for t in range(1, T_MAX + 1):  # Show first 10 time periods for brevity
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
        for t in range(1, T_MAX + 1):
            print(f"  t={t}: {s[t].solution_value():.2f}")

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
        #ax1.legend([f'Job {j}' for j in J], loc='upper right')

        # Plot energy storage
        ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
        battery_capacity = N_value * B
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
        plt.savefig('schedule_visualization_3_scip.svg', format="svg")
        print("\nSchedule visualization saved as 'schedule_visualization_3_scip.svg'")

        return M_value, N_value, solver.Objective().Value()

    else:
        print("Failed to find an optimal solution.")
        return None


if __name__ == "__main__":
    solve()
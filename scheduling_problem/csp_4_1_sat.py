from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
from combine_data import get_data

# Constants
M = 801
N = 1500

data = get_data()

def float_to_int_round(float_list):
    return (round(x) for x in float_list)

I = data["I"]
J = data["J"]
T = data["T"]
n_jobs = data["n_jobs"]
d = data["d"]
e = float_to_int_round(data["e"])
f = float_to_int_round(data["f"])
c_b = data["c_b"]
c_p = data["c_p"]
c = data["c"]
p = float_to_int_round(data["p"])
mmm = data["mmm"]
silent_periods = data["silent_periods"]
M_shared = data["M_shared"]
M_dependencies = data["M_dependencies"]
B = data["B"]
T_MAX = data["T_MAX"]
THRESHOLD_FOR_JOB_J_AND_I = data["THRESHOLD_FOR_JOB_J_AND_I"]
MACHINES = data["MACHINES"]



model = cp_model.CpModel()

# Variables
x = {}
y = {}
for i in I:
    for j in J:
        if j <= n_jobs[i]:
            for t in T:
                x[i, t, j] = model.NewBoolVar(f'x_{i}_{t}_{j}')
                y[i, t, j] = model.NewBoolVar(f'y_{i}_{t}_{j}')

s = {t: model.NewIntVar(0, N * B, f's_{t}') for t in T}

# Constraints
for t in T:
    model.Add(
        sum(e[i] * x[i, t, j] + f[i] * y[i, t, j] for i in I for j in J if j <= n_jobs[i]) <= M * p[t] + s[t]
    )

for t in T:
    if t == 1:
        model.Add(s[t] == 0)
    else:
        prev_sum = sum(
            M * p[tp] - sum(e[i] * x[i, tp, j] + f[i] * y[i, tp, j] for i in I for j in J if j <= n_jobs[i])
            for tp in range(1, t)
        )
        model.Add(s[t] <= prev_sum)

for t in T:
    model.Add(s[t] <= N * B)

for i in I:
    for j in J:
        if j <= n_jobs[i]:
            model.Add(sum(x[i, t, j] for t in T) == d[i])

for i in I:
    for t in T:
        model.Add(sum(x[i, t, j] for j in J if j <= n_jobs[i]) <= 1)
        model.Add(sum(y[i, t, j] for j in J if j <= n_jobs[i]) <= 1)

for t in T:
    model.Add(sum(e[i] * x[i, t, j] for i in I for j in J if j <= n_jobs[i]) <= mmm[t])

for i in I:
    for j in J:
        if j <= n_jobs[i]:
            for t in T:
                if t == 1:
                    model.Add(x[i, t, j] == y[i, t, j])
                else:
                    model.Add(x[i, t, j] <= y[i, t, j] + x[i, t - 1, j])
                model.Add(y[i, t, j] <= x[i, t, j])

for (k, kp1) in M_dependencies:
    if k in I and kp1 in I:
        for t in T:
            for j in J:
                if j <= n_jobs.get(k, 0) and j <= n_jobs.get(kp1, 0):
                    if t == 1:
                        model.Add(y[kp1, t, j] == 0)
                    else:
                        prev_completions = sum(x[k, tp, j] for tp in range(1, t))
                        model.Add(y[kp1, t, j] * d[k] <= prev_completions)

for i in I:
    for t in T:
        for j in J:
            if j <= n_jobs[i]:
                if t > c[i]:
                    cooldown_sum = sum(1 - sum(x[i, tp, jj] for jj in J if jj <= n_jobs[i]) for tp in range(t - c[i], t))
                    model.Add(y[i, t, j] * c[i] <= cooldown_sum)

for i in I:
    if i in silent_periods:
        for t in silent_periods[i]:
            if t <= T_MAX:
                model.Add(sum(x[i, t, j] for j in J if j <= n_jobs[i]) == 0)

for t in T:
    for group in M_shared:
        machines_in_group = [i for i in group if i in I]
        if machines_in_group:
            model.Add(sum(x[i, t, j] for i in machines_in_group for j in J if j <= n_jobs[i]) <= 1)

for i in I:
    for j in J:
        if j <= n_jobs[i]:
            for t in range(THRESHOLD_FOR_JOB_J_AND_I[(i, j)] + 1, T_MAX + 1):
                model.Add(x[i, t, j] == 0)

for i in I:
    for j in J:
        if j <= n_jobs[i]:
            model.Add(sum(y[i, t, j] for t in T) == 1)
            for t in range(2, T_MAX + 1):
                model.Add(x[i, t, j] - x[i, t - 1, j] <= y[i, t, j])

# Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    storage_values = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    for i in I:
        print(f"\nMachine {i} Schedule:")
        for j in J:
            if j <= n_jobs[i]:
                start_times = [t for t in T if solver.Value(y[i, t, j]) > 0]
                run_times = [t for t in T if solver.Value(x[i, t, j]) > 0]
                if start_times:
                    print(f"  Job {j} starts at: {start_times}")
                if run_times:
                    print(f"  Job {j} runs at: {run_times}")
                    for t in run_times:
                        ax1.bar(t, 1, bottom=i - 1, color=f'C{j}', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Machine')
    ax1.set_yticks(range(0, MACHINES))
    ax1.set_yticklabels([f'Machine {i}' for i in I])
    ax1.set_title('Machine Schedule')
    ax1.legend([f'Job {j}' for j in J], loc='upper right')

    for t in T:
        storage_values.append(solver.Value(s[t]))

    ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
    battery_capacity = N * B
    ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Energy Storage')
    ax2.set_title('Energy Storage Levels')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('schedule_visualization.png')
    print("\nSchedule visualization saved as 'schedule_visualization.png'")
    plt.show()
else:
    print("No feasible solution found.")

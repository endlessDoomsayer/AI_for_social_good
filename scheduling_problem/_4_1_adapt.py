import random
import math
import time
from copy import deepcopy
from combine_data import get_modified_data  # assuming you updated this with new constraints
import json


class LocalSearchRecovery:
    def __init__(self, data, M_val, N_val):
        self.data = data
        self.M_val = M_val
        self.N_val = N_val
        self.I = data["I"]
        self.J = data["J"]
        self.T = data["T"]
        self.T_MAX = data["T_MAX"]
        self.n_jobs = data["n_jobs"]
        self.d = data["d"]
        self.e = data["e"]
        self.f = data["f"]
        self.p = data["p"]
        self.THRESHOLD = data["THRESHOLD_FOR_JOB_J_AND_I"]

    def evaluate(self, solution):
        violations = 0
        for t in self.T:
            energy_used = 0
            for (i, j), start in solution.items():
                if start <= t <= start + self.d[i] - 1:
                    energy_used += self.e[i]
                if start == t:
                    energy_used += self.f[i]
            if energy_used > self.M_val * self.p[t]:
                violations += energy_used - self.M_val * self.p[t]
        return violations

    def get_neighbor(self, solution):
        neighbor = deepcopy(solution)
        i, j = random.choice(list(neighbor.keys()))
        valid_times = [t for t in self.T if t <= self.THRESHOLD.get((i, j), self.T_MAX)]
        if valid_times:
            neighbor[(i, j)] = random.choice(valid_times)
        return neighbor

    def simulated_annealing(self, solution, max_iter=10000, T_init=100, alpha=0.95):
        current = deepcopy(solution)
        best = deepcopy(solution)
        best_cost = current_cost = self.evaluate(current)
        T = T_init

        start_time = time.time()
        for _ in range(max_iter):
            neighbor = self.get_neighbor(current)
            cost = self.evaluate(neighbor)
            if cost < current_cost or random.random() < math.exp(-(cost - current_cost) / T):
                current = neighbor
                current_cost = cost
                if cost < best_cost:
                    best = deepcopy(neighbor)
                    best_cost = cost
            T *= alpha
            if best_cost == 0:
                break
        elapsed = time.time() - start_time
        return best, best_cost, elapsed

    def tabu_search(self, solution, max_iter=1000, tabu_size=50):
        current = deepcopy(solution)
        best = deepcopy(solution)
        best_cost = current_cost = self.evaluate(current)
        tabu_list = []
        start_time = time.time()

        for _ in range(max_iter):
            neighbors = [self.get_neighbor(current) for _ in range(20)]
            neighbors = [(n, self.evaluate(n)) for n in neighbors if n not in tabu_list]
            if not neighbors:
                continue
            neighbors.sort(key=lambda x: x[1])
            current, current_cost = neighbors[0]
            if current_cost < best_cost:
                best = deepcopy(current)
                best_cost = current_cost
            tabu_list.append(current)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
            if best_cost == 0:
                break
        elapsed = time.time() - start_time
        return best, best_cost, elapsed

    def run_all(self, original_solution):
        print("Running Simulated Annealing...")
        sa_result = self.simulated_annealing(original_solution)
        print(f"SA violations: {sa_result[1]}, time: {sa_result[2]:.2f}s")

        print("Running Tabu Search...")
        ts_result = self.tabu_search(original_solution)
        print(f"TS violations: {ts_result[1]}, time: {ts_result[2]:.2f}s")

        return {
            "Simulated Annealing": {"solution": sa_result[0], "violations": sa_result[1], "time": sa_result[2]},
            "Tabu Search": {"solution": ts_result[0], "violations": ts_result[1], "time": ts_result[2]},
        }

with open("optimal_schedule.json", "r") as f:
    loaded_schedule_raw = json.load(f)

# Original scheduling
original_solution = {tuple(map(int, k.split(","))): v for k, v in loaded_schedule_raw.items()}

data = get_modified_data()
M_val, N_val = 4912, 45

lsr = LocalSearchRecovery(data, M_val, N_val)
results = lsr.run_all(original_solution)
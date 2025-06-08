import random
import math
import time
from copy import deepcopy
from combine_data import get_modified_data
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
        self.c = data["c"]
        self.B = data["B"]
        self.mmm = data["mmm"]
        self.silent_periods = data["silent_periods"]
        self.M_shared = data["M_shared"]
        self.M_dependencies = data["M_dependencies"]
        self.THRESHOLD = data["THRESHOLD_FOR_JOB_J_AND_I"]

    def parse_solution(self, solution_dict):
        """Convert solution dictionary to proper format and compute y variables"""
        x = {}  # x[i,t,j] = 1 if machine i runs job j at time t
        y = {}  # y[i,t,j] = 1 if machine i starts job j at time t

        # Initialize all variables to 0
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        x[i, t, j] = 0
                        y[i, t, j] = 0

        # Parse the solution - format is "i,t,j": 1
        for key_str, value in solution_dict.items():
            if value == 1:
                parts = key_str.split(',')
                if len(parts) == 3:
                    i, t, j = map(int, parts)
                    if i in self.I and j <= self.n_jobs[i] and t in self.T:
                        x[i, t, j] = 1

        # Compute y variables based on x variables
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        if x[i, t, j] == 1:
                            # This is a start if it's the first time period or previous period was 0
                            if t == 1 or x.get((i, t - 1, j), 0) == 0:
                                y[i, t, j] = 1

        return x, y

    def solution_to_dict(self, x):
        """Convert x variables back to dictionary format"""
        solution_dict = {}
        for (i, t, j), value in x.items():
            if value == 1:
                solution_dict[f"{i},{t},{j}"] = 1
        return solution_dict

    def calculate_storage(self, x, y):
        """Calculate storage levels for all time periods - aligned with SCIP solver"""
        storage = {}
        for t in self.T:
            if t == 1:
                # Starting storage is 0
                storage[t] = 0
            else:
                storage[t] = storage[t - 1]
                storage[t] += self.M_val * self.p[t-1]

                # Subtract consumption from previous period (t-1)
                consumption_prev = sum(self.e[i] * x.get((i, t - 1, j), 0) +
                                       self.f[i] * y.get((i, t - 1, j), 0)
                                       for i in self.I for j in self.J if j <= self.n_jobs[i])
                storage[t] -= consumption_prev

        return storage

    def evaluate_all_constraints(self, x, y, first=False):
        """Evaluate all constraint violations with proper weights"""
        violations = 0
        violation_details = {}

        # Calculate storage levels
        storage = self.calculate_storage(x, y)

        # 1. Storage constraints (energy balance and battery capacity)
        storage_violations = 0
        for t in self.T:
            # Storage cannot be negative
            if storage[t] < 0:
                if first:
                    print("Negativeeee at time ", t, " there is storage ", storage[t])
                storage_violations += abs(storage[t]) * 10  # High penalty
            # Storage cannot exceed battery capacity
            if storage[t] > self.N_val * self.B:
                if first:
                    print("Too muchhhh")
                storage_violations += (storage[t] - self.N_val * self.B) * 10

        violation_details['storage'] = storage_violations
        violations += storage_violations

        # 2. Job duration constraint violations
        duration_violations = 0
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    total_runtime = sum(x.get((i, t, j), 0) for t in self.T)
                    if total_runtime != self.d[i]:
                        duration_violations += abs(total_runtime - self.d[i]) * 100  # Very high penalty

        violation_details['duration'] = duration_violations
        violations += duration_violations

        # 3. Machine capacity violations (one job at a time)
        capacity_violations = 0
        for i in self.I:
            for t in self.T:
                running_jobs = sum(x.get((i, t, j), 0) for j in self.J if j <= self.n_jobs[i])
                starting_jobs = sum(y.get((i, t, j), 0) for j in self.J if j <= self.n_jobs[i])
                if running_jobs > 1:
                    capacity_violations += (running_jobs - 1) * 50
                if starting_jobs > 1:
                    capacity_violations += (starting_jobs - 1) * 50

        violation_details['capacity'] = capacity_violations
        violations += capacity_violations

        # 4. Maximum energy per time period violations
        max_energy_violations = 0
        for t in self.T:
            total_energy = sum(self.e[i] * x.get((i, t, j), 0)
                               for i in self.I for j in self.J if j <= self.n_jobs[i])
            if total_energy > self.mmm[t]:
                max_energy_violations += (total_energy - self.mmm[t]) * 20

        violation_details['max_energy'] = max_energy_violations
        violations += max_energy_violations

        # 5. Silent period violations
        silent_violations = 0
        for i in self.I:
            if i in self.silent_periods:
                for t in self.silent_periods[i]:
                    if t <= self.T_MAX and t in self.T:
                        running_jobs = sum(x.get((i, t, j), 0) for j in self.J if j <= self.n_jobs[i])
                        silent_violations += running_jobs * 100

        violation_details['silent'] = silent_violations
        violations += silent_violations

        # 6. Shared machine violations
        shared_violations = 0
        for t in self.T:
            for group in self.M_shared:
                machines_in_group = [i for i in group if i in self.I]
                if machines_in_group:
                    total_usage = sum(x.get((i, t, j), 0)
                                      for i in machines_in_group
                                      for j in self.J if j <= self.n_jobs[i])
                    if total_usage > 1:
                        shared_violations += (total_usage - 1) * 50

        violation_details['shared'] = shared_violations
        violations += shared_violations

        # 7. Dependency constraints
        dependency_violations = 0
        for (k, kp1) in self.M_dependencies:
            if k in self.I and kp1 in self.I:
                for j in self.J:
                    if j <= self.n_jobs.get(k, 0) and j <= self.n_jobs.get(kp1, 0):
                        for t in self.T:
                            if y.get((kp1, t, j), 0) == 1:  # kp1 starts job j at time t
                                if t == 1:
                                    dependency_violations += 100  # Cannot start at t=1
                                else:
                                    # Check if k has completed d[k] time units of job j before time t
                                    completed_k = sum(x.get((k, tp, j), 0) for tp in range(1, t))
                                    if completed_k < self.d[k]:
                                        dependency_violations += (self.d[k] - completed_k) * 50

        violation_details['dependency'] = dependency_violations
        violations += dependency_violations

        # 8. Cooldown constraint violations
        cooldown_violations = 0
        for i in self.I:
            for t in self.T:
                if t > self.c[i]:
                    for j in self.J:
                        starts_at_t = y.get((i, t, j), 0)
                        if starts_at_t:
                            runs_before = sum(x.get((i, tp, j), 0)
                                              for tp in range(t - self.c[i], t))

                            if runs_before > 0:
                                cooldown_violations += 30

        violation_details['cooldown'] = cooldown_violations
        violations += cooldown_violations

        # 9. Threshold violations
        threshold_violations = 0
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    threshold = self.THRESHOLD.get((i, j), self.T_MAX)
                    for t in range(threshold + 1, self.T_MAX + 1):
                        if t in self.T and x.get((i, t, j), 0) > 0:
                            threshold_violations += 100

        violation_details['threshold'] = threshold_violations
        violations += threshold_violations

        # 10. Job continuity violations (start implies run, continuity)
        continuity_violations = 0
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        # Start implies run
                        if y.get((i, t, j), 0) > x.get((i, t, j), 0):
                            continuity_violations += 50

                        # Continuity constraint
                        if t == 1:
                            if x.get((i, t, j), 0) != y.get((i, t, j), 0):
                                continuity_violations += 50
                        else:
                            if x.get((i, t, j), 0) > y.get((i, t, j), 0) + x.get((i, t - 1, j), 0):
                                continuity_violations += 50

        violation_details['continuity'] = continuity_violations
        violations += continuity_violations

        return violations, violation_details

    def evaluate(self, solution_dict, first = False):
        """Evaluate constraint violations for a solution"""
        x, y = self.parse_solution(solution_dict)
        violations, details = self.evaluate_all_constraints(x, y, first=first)
        if (first):
            print(details)
        return violations

    def get_neighbor_move_job(self, solution_dict):
        """Move a single job execution to a different time period"""
        neighbor = deepcopy(solution_dict)

        if not neighbor:
            return neighbor

        # Pick a random job assignment to move
        key = random.choice(list(neighbor.keys()))
        parts = key.split(',')
        if len(parts) != 3:
            return neighbor

        i, old_t, j = map(int, parts)

        # Find valid time periods for this job
        threshold = self.THRESHOLD.get((i, j), self.T_MAX)
        valid_times = [t for t in self.T if t <= threshold and t != old_t]

        if not valid_times:
            return neighbor

        new_t = random.choice(valid_times)

        # Remove old assignment and add new one
        del neighbor[key]
        new_key = f"{i},{new_t},{j}"
        neighbor[new_key] = 1

        return neighbor

    def get_neighbor_swap_times(self, solution_dict):
        """Swap execution times of two jobs"""
        neighbor = deepcopy(solution_dict)

        if len(neighbor) < 2:
            return neighbor

        keys = list(neighbor.keys())
        key1, key2 = random.sample(keys, 2)

        parts1 = key1.split(',')
        parts2 = key2.split(',')

        if len(parts1) != 3 or len(parts2) != 3:
            return neighbor

        i1, t1, j1 = map(int, parts1)
        i2, t2, j2 = map(int, parts2)

        # Check if swap is valid (respects thresholds)
        threshold1 = self.THRESHOLD.get((i1, j1), self.T_MAX)
        threshold2 = self.THRESHOLD.get((i2, j2), self.T_MAX)

        if t2 <= threshold1 and t1 <= threshold2:
            del neighbor[key1]
            del neighbor[key2]
            neighbor[f"{i1},{t2},{j1}"] = 1
            neighbor[f"{i2},{t1},{j2}"] = 1

        return neighbor

    def get_neighbor_shift_job_sequence(self, solution_dict):
        """Shift an entire job sequence (all time periods of a job) to start earlier/later"""
        neighbor = deepcopy(solution_dict)

        # Group assignments by (machine, job)
        job_assignments = {}
        for key, value in neighbor.items():
            if value == 1:
                parts = key.split(',')
                if len(parts) == 3:
                    i, t, j = map(int, parts)
                    if (i, j) not in job_assignments:
                        job_assignments[(i, j)] = []
                    job_assignments[(i, j)].append(t)

        if not job_assignments:
            return neighbor

        # Pick a random job to shift
        (i, j), time_periods = random.choice(list(job_assignments.items()))
        time_periods.sort()

        # Calculate possible shifts
        threshold = self.THRESHOLD.get((i, j), self.T_MAX)
        min_start = min(self.T)
        max_end = min(threshold, max(self.T))
        job_duration = len(time_periods)

        # Possible start times that keep job within bounds
        possible_starts = [t for t in range(min_start, max_end - job_duration + 2)
                           if t + job_duration - 1 <= max_end]

        if len(possible_starts) <= 1:
            return neighbor

        current_start = min(time_periods)
        new_starts = [s for s in possible_starts if s != current_start]

        if not new_starts:
            return neighbor

        new_start = random.choice(new_starts)
        shift = new_start - current_start

        # Remove old assignments
        for t in time_periods:
            old_key = f"{i},{t},{j}"
            if old_key in neighbor:
                del neighbor[old_key]

        # Add new assignments
        for t in time_periods:
            new_t = t + shift
            if new_t in self.T:
                neighbor[f"{i},{new_t},{j}"] = 1

        return neighbor

    def get_neighbor(self, solution_dict):
        """Generate a neighboring solution using various strategies"""
        strategy = random.choice(['move', 'swap', 'shift'])

        if strategy == 'move':
            return self.get_neighbor_move_job(solution_dict)
        elif strategy == 'swap':
            return self.get_neighbor_swap_times(solution_dict)
        else:  # shift
            return self.get_neighbor_shift_job_sequence(solution_dict)

    def simulated_annealing(self, solution, max_iter=50000, T_init=1000, alpha=0.99):
        """Simulated Annealing algorithm"""
        current = deepcopy(solution)
        best = deepcopy(solution)
        best_cost = current_cost = self.evaluate(current, first=True)
        T = T_init

        print(f"SA Initial cost: {current_cost}")

        start_time = time.time()
        for iteration in range(max_iter):
            neighbor = self.get_neighbor(current)
            cost = self.evaluate(neighbor)

            # Accept or reject the neighbor
            if cost < current_cost or (T > 0 and random.random() < math.exp(-(cost - current_cost) / T)):
                current = neighbor
                current_cost = cost

                if cost < best_cost:
                    best = deepcopy(neighbor)
                    best_cost = cost
                    print(f"SA Iteration {iteration}: New best cost = {best_cost}")

            T *= alpha

            if best_cost == 0:
                print(f"SA: Found feasible solution at iteration {iteration}")
                break

            if iteration % 5000 == 0:
                print(f"SA Iteration {iteration}: Current cost = {current_cost}, Best cost = {best_cost}, T = {T:.4f}")

        elapsed = time.time() - start_time
        print(f"SA completed in {elapsed:.2f}s")
        return best, best_cost, elapsed

    def tabu_search(self, solution, max_iter=10000, tabu_size=100):
        """Tabu Search algorithm"""
        current = deepcopy(solution)
        best = deepcopy(solution)
        best_cost = current_cost = self.evaluate(current)
        tabu_list = []

        print(f"TS Initial cost: {current_cost}")

        start_time = time.time()

        for iteration in range(max_iter):
            # Generate multiple neighbors
            neighbors = []
            for _ in range(30):  # Generate more neighbors
                neighbor = self.get_neighbor(current)
                neighbor_key = str(sorted(neighbor.items()))
                if neighbor_key not in tabu_list:
                    neighbors.append((neighbor, self.evaluate(neighbor), neighbor_key))

            # If no non-tabu neighbors, generate some anyway (aspiration criteria)
            if not neighbors:
                for _ in range(10):
                    neighbor = self.get_neighbor(current)
                    neighbor_key = str(sorted(neighbor.items()))
                    neighbors.append((neighbor, self.evaluate(neighbor), neighbor_key))

            if not neighbors:
                break

            # Sort by cost and pick the best
            neighbors.sort(key=lambda x: x[1])
            current, current_cost, current_key = neighbors[0]

            if current_cost < best_cost:
                best = deepcopy(current)
                best_cost = current_cost
                print(f"TS Iteration {iteration}: New best cost = {best_cost}")

            # Add current solution to tabu list
            tabu_list.append(current_key)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

            if best_cost == 0:
                print(f"TS: Found feasible solution at iteration {iteration}")
                break

            if iteration % 1000 == 0:
                print(f"TS Iteration {iteration}: Current cost = {current_cost}, Best cost = {best_cost}")

        elapsed = time.time() - start_time
        print(f"TS completed in {elapsed:.2f}s")
        return best, best_cost, elapsed

    def run_all(self, original_solution):
        """Run both algorithms and return results"""
        print("Initial solution evaluation...")
        initial_cost = self.evaluate(original_solution)
        print(f"Initial violations: {initial_cost}")

        # Get detailed violation breakdown
        x, y = self.parse_solution(original_solution)
        _, details = self.evaluate_all_constraints(x, y)
        print(f"Violation breakdown: {details}")

        if initial_cost == 0:
            print("Original solution is already feasible!")
            return {
                "Original": {"solution": original_solution, "violations": initial_cost, "time": 0},
                "Simulated Annealing": {"solution": original_solution, "violations": initial_cost, "time": 0},
                "Tabu Search": {"solution": original_solution, "violations": initial_cost, "time": 0},
            }

        print("\nRunning Simulated Annealing...")
        sa_result = self.simulated_annealing(original_solution)
        print(f"SA final violations: {sa_result[1]}, time: {sa_result[2]:.2f}s")

        print("\nRunning Tabu Search...")
        ts_result = self.tabu_search(original_solution)
        print(f"TS final violations: {ts_result[1]}, time: {ts_result[2]:.2f}s")

        return {
            "Original": {"solution": original_solution, "violations": initial_cost, "time": 0},
            "Simulated Annealing": {"solution": sa_result[0], "violations": sa_result[1], "time": sa_result[2]},
            "Tabu Search": {"solution": ts_result[0], "violations": ts_result[1], "time": ts_result[2]},
        }


def solve(M, N, data=get_modified_data()):
    """Main solving function"""
    print(f"Loading solution and running local search for M={M}, N={N}")

    try:
        with open("optimal_schedule.json", "r") as f:
            loaded_schedule_raw = json.load(f)

        print(f"Loaded {len(loaded_schedule_raw)} job assignments")

        # The JSON format is already correct - no need to convert tuple keys
        original_solution = loaded_schedule_raw

        lsr = LocalSearchRecovery(data, M, N)
        results = lsr.run_all(original_solution)

        # Save best results
        best_method = min(results.keys(), key=lambda k: results[k]["violations"])
        print(f"\nBest method: {best_method} with {results[best_method]['violations']} violations")

        if results[best_method]["violations"] == 0:
            print("Found feasible solution!")
            with open("improved_schedule.json", "w") as f:
                json.dump(results[best_method]["solution"], f, indent=2)
            print("Saved improved solution to 'improved_schedule.json'")
        else:
            print("Could not find a completely feasible solution.")
            print("Saving best solution found...")
            with open("best_schedule.json", "w") as f:
                json.dump(results[best_method]["solution"], f, indent=2)

        return results

    except FileNotFoundError:
        print("optimal_schedule.json not found. Please run the CSP solver first.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    solve(2491, 166)
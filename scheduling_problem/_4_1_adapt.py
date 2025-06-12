import random
import math
import time
from copy import deepcopy
from combine_data import get_modified_data
import json
from collections import defaultdict


class EnhancedLocalSearchRecovery:
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
        x = {}
        y = {}

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
        """Calculate storage levels for all time periods"""
        storage = {}
        for t in self.T:
            if t == 1:
                storage[t] = 0
            else:
                storage[t] = storage[t - 1]
                storage[t] += self.M_val * self.p[t - 1]

                consumption_prev = sum(self.e[i] * x.get((i, t - 1, j), 0) +
                                       self.f[i] * y.get((i, t - 1, j), 0)
                                       for i in self.I for j in self.J if j <= self.n_jobs[i])
                storage[t] -= consumption_prev

        return storage

    def evaluate_all_constraints(self, x, y):
        """Evaluate all constraint violations with proper weights"""
        violations = 0
        violation_details = {}

        # 1. Storage constraints can't be violated since we didn't change panels
        storage_violations = 0
        storage = self.calculate_storage(x,y)
        for t in self.T:
            if storage[t] >= self.N_val*self.B:
                storage_violations += (storage[t]-self.N_val*self.B) * 50

        violation_details['storage'] = storage_violations
        violations += storage_violations

        # 2. Job duration constraint violations
        duration_violations = 0
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    total_runtime = sum(x.get((i, t, j), 0) for t in self.T)
                    if total_runtime != self.d[i]:
                        duration_violations += abs(total_runtime - self.d[i]) * 100

        violation_details['duration'] = duration_violations
        violations += duration_violations

        # 3. Machine capacity violations
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
                            if y.get((kp1, t, j), 0) == 1:
                                if t == 1:
                                    dependency_violations += 100
                                else:
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

        # 10. Job continuity violations
        continuity_violations = 0
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        if y.get((i, t, j), 0) > x.get((i, t, j), 0):
                            continuity_violations += 50

                        if t == 1:
                            if x.get((i, t, j), 0) != y.get((i, t, j), 0):
                                continuity_violations += 50
                        else:
                            if x.get((i, t, j), 0) > y.get((i, t, j), 0) + x.get((i, t - 1, j), 0):
                                continuity_violations += 50

        violation_details['continuity'] = continuity_violations
        violations += continuity_violations

        return violations, violation_details

    def evaluate(self, solution_dict):
        """Evaluate constraint violations for a solution"""
        x, y = self.parse_solution(solution_dict)
        violations, details = self.evaluate_all_constraints(x, y)
        return violations

    def get_job_schedule(self, solution_dict):
        """Get current job schedules organized by machine and job"""
        job_schedules = defaultdict(list)
        for key, value in solution_dict.items():
            if value == 1:
                parts = key.split(',')
                if len(parts) == 3:
                    i, t, j = map(int, parts)
                    job_schedules[(i, j)].append(t)

        for key in job_schedules:
            job_schedules[key].sort()

        return job_schedules

    def repair_job_duration(self, solution_dict):
        """Repair jobs that don't have the correct duration"""
        neighbor = deepcopy(solution_dict)
        job_schedules = self.get_job_schedule(neighbor)

        for (i, j), time_periods in job_schedules.items():
            required_duration = self.d[i]
            current_duration = len(time_periods)

            if current_duration != required_duration:
                for t in time_periods:
                    key = f"{i},{t},{j}"
                    if key in neighbor:
                        del neighbor[key]

                threshold = self.THRESHOLD.get((i, j), self.T_MAX)
                valid_start_times = [t for t in self.T
                                     if t + required_duration - 1 <= min(threshold, max(self.T))]

                if valid_start_times:
                    start_time = random.choice(valid_start_times)
                    for offset in range(required_duration):
                        new_t = start_time + offset
                        if new_t in self.T:
                            neighbor[f"{i},{new_t},{j}"] = 1

        return neighbor

    def get_neighbor_smart_repair(self, solution_dict):
        """Smart repair that focuses on the most violated constraints"""
        x, y = self.parse_solution(solution_dict)
        violations, details = self.evaluate_all_constraints(x, y)

        max_violation_type = max(details.keys(), key=lambda k: details[k])

        if max_violation_type == 'duration':
            return self.repair_job_duration(solution_dict)
        elif max_violation_type == 'storage':
            return self.get_neighbor_reduce_energy_consumption(solution_dict)
        elif max_violation_type == 'capacity':
            return self.get_neighbor_resolve_conflicts(solution_dict)
        else:
            return self.get_neighbor_move_job(solution_dict)

    def get_neighbor_reduce_energy_consumption(self, solution_dict):
        """Move jobs to reduce energy consumption in overloaded periods"""
        neighbor = deepcopy(solution_dict)
        x, y = self.parse_solution(neighbor)
        storage = self.calculate_storage(x, y)

        problem_periods = []
        for t in self.T:
            if storage[t] < 0:
                problem_periods.append(t)

        if not problem_periods:
            return self.get_neighbor_move_job(neighbor)

        problem_period = random.choice(problem_periods)
        jobs_in_period = []

        for key, value in neighbor.items():
            if value == 1:
                parts = key.split(',')
                if len(parts) == 3:
                    i, t, j = map(int, parts)
                    if t == problem_period:
                        jobs_in_period.append((i, t, j, key))

        if jobs_in_period:
            i, old_t, j, old_key = random.choice(jobs_in_period)
            threshold = self.THRESHOLD.get((i, j), self.T_MAX)

            alternative_times = [t for t in self.T
                                 if t != old_t and t <= threshold and storage.get(t, 0) >= 0]

            if alternative_times:
                new_t = random.choice(alternative_times)
                del neighbor[old_key]
                neighbor[f"{i},{new_t},{j}"] = 1

        return neighbor

    def get_neighbor_resolve_conflicts(self, solution_dict):
        """Resolve machine capacity conflicts"""
        neighbor = deepcopy(solution_dict)

        machine_usage = defaultdict(lambda: defaultdict(list))
        for key, value in neighbor.items():
            if value == 1:
                parts = key.split(',')
                if len(parts) == 3:
                    i, t, j = map(int, parts)
                    machine_usage[i][t].append((j, key))

        conflicts = []
        for i, time_jobs in machine_usage.items():
            for t, jobs in time_jobs.items():
                if len(jobs) > 1:
                    conflicts.append((i, t, jobs))

        if conflicts:
            i, t, jobs = random.choice(conflicts)
            j, old_key = random.choice(jobs)
            threshold = self.THRESHOLD.get((i, j), self.T_MAX)

            alternative_times = [new_t for new_t in self.T
                                 if new_t != t and new_t <= threshold]

            if alternative_times:
                new_t = random.choice(alternative_times)
                del neighbor[old_key]
                neighbor[f"{i},{new_t},{j}"] = 1

        return neighbor

    def get_neighbor_move_job(self, solution_dict):
        """Move a single job execution to a different time period"""
        neighbor = deepcopy(solution_dict)

        if not neighbor:
            return neighbor

        key = random.choice(list(neighbor.keys()))
        parts = key.split(',')
        if len(parts) != 3:
            return neighbor

        i, old_t, j = map(int, parts)
        threshold = self.THRESHOLD.get((i, j), self.T_MAX)
        valid_times = [t for t in self.T if t <= threshold and t != old_t]

        if not valid_times:
            return neighbor

        new_t = random.choice(valid_times)
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

        threshold1 = self.THRESHOLD.get((i1, j1), self.T_MAX)
        threshold2 = self.THRESHOLD.get((i2, j2), self.T_MAX)

        if t2 <= threshold1 and t1 <= threshold2:
            del neighbor[key1]
            del neighbor[key2]
            neighbor[f"{i1},{t2},{j1}"] = 1
            neighbor[f"{i2},{t1},{j2}"] = 1

        return neighbor

    def get_neighbor_shift_job_sequence(self, solution_dict):
        """Shift an entire job sequence to start earlier/later"""
        neighbor = deepcopy(solution_dict)

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

        (i, j), time_periods = random.choice(list(job_assignments.items()))
        time_periods.sort()

        threshold = self.THRESHOLD.get((i, j), self.T_MAX)
        min_start = min(self.T)
        max_end = min(threshold, max(self.T))
        job_duration = len(time_periods)

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

        for t in time_periods:
            old_key = f"{i},{t},{j}"
            if old_key in neighbor:
                del neighbor[old_key]

        for t in time_periods:
            new_t = t + shift
            if new_t in self.T:
                neighbor[f"{i},{new_t},{j}"] = 1

        return neighbor

    def get_neighbor(self, solution_dict):
        """Generate a neighboring solution using various strategies"""
        strategy = random.choice(['smart_repair', 'move', 'swap', 'shift'])

        if strategy == 'smart_repair':
            return self.get_neighbor_smart_repair(solution_dict)
        elif strategy == 'move':
            return self.get_neighbor_move_job(solution_dict)
        elif strategy == 'swap':
            return self.get_neighbor_swap_times(solution_dict)
        else:  # shift
            return self.get_neighbor_shift_job_sequence(solution_dict)

    def adaptive_simulated_annealing(self, solution, max_iter=100000, T_init=2000, alpha=0.995, reheat_factor=1.5):
        """Enhanced Simulated Annealing with adaptive temperature and reheating"""
        current = deepcopy(solution)
        best = deepcopy(solution)
        best_cost = current_cost = self.evaluate(current)
        T = T_init

        # Adaptive parameters
        no_improvement_count = 0
        reheat_threshold = max_iter // 10

        print(f"ASA Initial cost: {current_cost}")

        start_time = time.time()
        for iteration in range(max_iter):
            neighbor = self.get_neighbor(current)
            cost = self.evaluate(neighbor)

            # Accept or reject the neighbor
            if cost < current_cost:
                current = neighbor
                current_cost = cost
                no_improvement_count = 0

                if cost < best_cost:
                    best = deepcopy(neighbor)
                    best_cost = cost
                    print(f"ASA Iteration {iteration}: New best cost = {best_cost}")
            elif T > 0 and random.random() < math.exp(-(cost - current_cost) / T):
                current = neighbor
                current_cost = cost
                no_improvement_count += 1
            else:
                no_improvement_count += 1

            if no_improvement_count < 100:
                T *= alpha
            else:
                T *= alpha * 0.99

            if no_improvement_count > reheat_threshold and T < T_init * 0.01:
                T *= reheat_factor
                no_improvement_count = 0
                print(f"ASA Iteration {iteration}: Reheating to T = {T:.4f}")

            if best_cost == 0:
                print(f"ASA: Found feasible solution at iteration {iteration}")
                break

            if iteration % 10000 == 0:
                print(f"ASA Iteration {iteration}: Current cost = {current_cost}, Best cost = {best_cost}, T = {T:.4f}")

        elapsed = time.time() - start_time
        print(f"ASA completed in {elapsed:.2f}s")
        return best, best_cost, elapsed

    def enhanced_tabu_search(self, solution, max_iter=50000, tabu_size=200, intensification_threshold=1000):
        """Enhanced Tabu Search with intensification and diversification"""
        current = deepcopy(solution)
        best = deepcopy(solution)
        best_cost = current_cost = self.evaluate(current)
        tabu_list = []
        frequency = defaultdict(int)
        no_improvement_count = 0

        print(f"ETS Initial cost: {current_cost}")

        start_time = time.time()

        for iteration in range(max_iter):
            neighbors = []
            neighbor_count = 50 if no_improvement_count > intensification_threshold else 30

            for _ in range(neighbor_count):
                if no_improvement_count > intensification_threshold:
                    neighbor = self.get_neighbor_with_diversification(current, frequency)
                else:
                    neighbor = self.get_neighbor(current)

                neighbor_key = str(sorted(neighbor.items()))
                if neighbor_key not in tabu_list:
                    cost = self.evaluate(neighbor)
                    neighbors.append((neighbor, cost, neighbor_key))

            if not neighbors:
                for _ in range(20):
                    neighbor = self.get_neighbor(current)
                    neighbor_key = str(sorted(neighbor.items()))
                    cost = self.evaluate(neighbor)
                    if cost < best_cost:  # Aspiration criteria
                        neighbors.append((neighbor, cost, neighbor_key))

            if not neighbors:
                break

            neighbors.sort(key=lambda x: x[1])
            current, current_cost, current_key = neighbors[0]

            frequency[current_key] += 1

            if current_cost < best_cost:
                best = deepcopy(current)
                best_cost = current_cost
                no_improvement_count = 0
                print(f"ETS Iteration {iteration}: New best cost = {best_cost}")
            else:
                no_improvement_count += 1

            tabu_list.append(current_key)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

            if best_cost == 0:
                print(f"ETS: Found feasible solution at iteration {iteration}")
                break

            if iteration % 5000 == 0:
                print(f"ETS Iteration {iteration}: Current cost = {current_cost}, Best cost = {best_cost}")

        elapsed = time.time() - start_time
        print(f"ETS completed in {elapsed:.2f}s")
        return best, best_cost, elapsed

    def get_neighbor_with_diversification(self, solution_dict, frequency):
        """Generate neighbor with preference for less frequent moves"""
        candidates = []
        for _ in range(5):
            neighbor = self.get_neighbor(solution_dict)
            neighbor_key = str(sorted(neighbor.items()))
            candidates.append((neighbor, frequency.get(neighbor_key, 0)))

        return min(candidates, key=lambda x: x[1])[0]

    def run_all(self, original_solution):
        """Run both enhanced algorithms and return results"""
        print("Initial solution evaluation...")
        initial_cost = self.evaluate(original_solution)
        print(f"Initial violations: {initial_cost}")

        x, y = self.parse_solution(original_solution)
        _, details = self.evaluate_all_constraints(x, y)
        print(f"Violation breakdown: {details}")

        if initial_cost == 0:
            print("Original solution is already feasible!")
            return {
                "Original": {"solution": original_solution, "violations": initial_cost, "time": 0},
                "Adaptive SA": {"solution": original_solution, "violations": initial_cost, "time": 0},
                "Enhanced TS": {"solution": original_solution, "violations": initial_cost, "time": 0},
            }

        print("\nRunning Adaptive Simulated Annealing...")
        asa_result = self.adaptive_simulated_annealing(original_solution)
        print(f"ASA final violations: {asa_result[1]}, time: {asa_result[2]:.2f}s")

        print("\nRunning Enhanced Tabu Search...")
        ets_result = self.enhanced_tabu_search(original_solution)
        print(f"ETS final violations: {ets_result[1]}, time: {ets_result[2]:.2f}s")

        return {
            "Original": {"solution": original_solution, "violations": initial_cost, "time": 0},
            "Adaptive SA": {"solution": asa_result[0], "violations": asa_result[1], "time": asa_result[2]},
            "Enhanced TS": {"solution": ets_result[0], "violations": ets_result[1], "time": ets_result[2]},
        }


def solve_enhanced(M, N, data=get_modified_data()):
    """Enhanced solving function"""
    print(f"Loading solution and running enhanced local search for M={M}, N={N}")

    try:
        with open("output_1day/optimal_schedule.json", "r") as f:
            loaded_schedule_raw = json.load(f)

        print(f"Loaded {len(loaded_schedule_raw)} job assignments")

        original_solution = loaded_schedule_raw

        lsr = EnhancedLocalSearchRecovery(data, M, N)
        results = lsr.run_all(original_solution)

        # Save best results
        best_method = min(results.keys(), key=lambda k: results[k]["violations"])
        print(f"\nBest method: {best_method} with {results[best_method]['violations']} violations")

        if results[best_method]["violations"] == 0:
            print("Found feasible solution!")
            with open("output_1day/improved_schedule.json", "w") as f:
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
    solve_enhanced(2491, 166)
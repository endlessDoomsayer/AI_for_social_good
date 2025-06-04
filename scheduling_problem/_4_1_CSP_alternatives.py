from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict
from combine_data import get_data
import time
import json


# THIS COMPARES THE STANDARD CSP with other enhanced techniques

def float_to_int_round(float_list):
    return {x: round(float_list[x]) for x in float_list}

def get_activity_times(solver, x):
    """
    Extract job start times from solution.
    Returns: dict (i, j) -> start_time
    """
    activity_times = {}
    for (i, t, j), var in x.items():
        if solver.BooleanValue(var):
            activity_times[(i, j)] = t
    return activity_times


class CSPSolver:
    def __init__(self, data):
        self.data = data
        self.I = data["I"]
        self.J = data["J"]
        self.T = data["T"]
        self.n_jobs = data["n_jobs"]
        self.d = data["d"]
        self.e = float_to_int_round(data["e"])
        self.f = float_to_int_round(data["f"])
        self.c_b = data["c_b"]
        self.c_p = data["c_p"]
        self.c = data["c"]
        self.p = float_to_int_round(data["p"])
        self.mmm = data["mmm"]
        self.silent_periods = data["silent_periods"]
        self.M_shared = data["M_shared"]
        self.M_dependencies = data["M_dependencies"]
        self.B = data["B"]
        self.T_MAX = data["T_MAX"]
        self.THRESHOLD_FOR_JOB_J_AND_I = data["THRESHOLD_FOR_JOB_J_AND_I"]
        self.MACHINES = data["MACHINES"]

    def create_base_model(self, M_val, N_val):
        """Create the base CP model with all constraints"""
        model = cp_model.CpModel()

        # Variables
        x = {}
        y = {}
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        x[i, t, j] = model.NewBoolVar(f'x_{i}_{t}_{j}')
                        y[i, t, j] = model.NewBoolVar(f'y_{i}_{t}_{j}')

        s = {t: model.NewIntVar(0, N_val * self.B, f's_{t}') for t in self.T}

        # All constraints from original model
        self._add_all_constraints(model, x, y, s, M_val, N_val)

        return model, x, y, s

    def _add_all_constraints(self, model, x, y, s, M_val, N_val):
        """Add all constraints to the model"""
        # Energy constraints
        for t in self.T:
            model.Add(
                sum(self.e[i] * x[i, t, j] + self.f[i] * y[i, t, j]
                    for i in self.I for j in self.J if j <= self.n_jobs[i]) <= M_val * self.p[t] + s[t]
            )

        # Storage constraints
        for t in self.T:
            if t == 1:
                model.Add(s[t] == 0)
            else:
                prev_sum = sum(
                    M_val * self.p[tp] - sum(self.e[i] * x[i, tp, j] + self.f[i] * y[i, tp, j]
                                             for i in self.I for j in self.J if j <= self.n_jobs[i])
                    for tp in range(1, t)
                )
                model.Add(s[t] <= prev_sum)

        for t in self.T:
            model.Add(s[t] <= N_val * self.B)

        # Job completion constraints
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    model.Add(sum(x[i, t, j] for t in self.T) == self.d[i])

        # Machine capacity constraints
        for i in self.I:
            for t in self.T:
                model.Add(sum(x[i, t, j] for j in self.J if j <= self.n_jobs[i]) <= 1)
                model.Add(sum(y[i, t, j] for j in self.J if j <= self.n_jobs[i]) <= 1)

        # Maximum energy per time period
        for t in self.T:
            model.Add(sum(self.e[i] * x[i, t, j] for i in self.I for j in self.J if j <= self.n_jobs[i]) <= self.mmm[t])

        # Job continuity constraints
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        if t == 1:
                            model.Add(x[i, t, j] == y[i, t, j])
                        else:
                            model.Add(x[i, t, j] <= y[i, t, j] + x[i, t - 1, j])
                        model.Add(y[i, t, j] <= x[i, t, j])

        # Dependencies
        for (k, kp1) in self.M_dependencies:
            if k in self.I and kp1 in self.I:
                for t in self.T:
                    for j in self.J:
                        if j <= self.n_jobs.get(k, 0) and j <= self.n_jobs.get(kp1, 0):
                            if t == 1:
                                model.Add(y[kp1, t, j] == 0)
                            else:
                                prev_completions = sum(x[k, tp, j] for tp in range(1, t))
                                model.Add(y[kp1, t, j] * self.d[k] <= prev_completions)

        # Cooldown constraints
        for i in self.I:
            for t in self.T:
                for j in self.J:
                    if j <= self.n_jobs[i]:
                        if t > self.c[i]:
                            cooldown_sum = sum(1 - sum(x[i, tp, jj] for jj in self.J if jj <= self.n_jobs[i])
                                               for tp in range(t - self.c[i], t))
                            model.Add(y[i, t, j] * self.c[i] <= cooldown_sum)

        # Silent periods
        for i in self.I:
            if i in self.silent_periods:
                for t in self.silent_periods[i]:
                    if t <= self.T_MAX:
                        model.Add(sum(x[i, t, j] for j in self.J if j <= self.n_jobs[i]) == 0)

        # Shared machine constraints
        for t in self.T:
            for group in self.M_shared:
                machines_in_group = [i for i in group if i in self.I]
                if machines_in_group:
                    model.Add(sum(x[i, t, j] for i in machines_in_group for j in self.J if j <= self.n_jobs[i]) <= 1)

        # Threshold constraints
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in range(self.THRESHOLD_FOR_JOB_J_AND_I[(i, j)] + 1, self.T_MAX + 1):
                        model.Add(x[i, t, j] == 0)

        # Job start constraints
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    model.Add(sum(y[i, t, j] for t in self.T) == 1)
                    for t in range(2, self.T_MAX + 1):
                        model.Add(x[i, t, j] - x[i, t - 1, j] <= y[i, t, j])

    def standard_backtracking_solver(self, M_val, N_val):
        """Standard backtracking"""
        print(f"Using Advanced Backtracking with MCV, LCV, Backjumping and No-good Learning for M={M_val}, N={N_val}")

        model = cp_model.CpModel()

        x = {}
        y = {}
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        x[i, t, j] = model.NewBoolVar(f'x_{i}_{t}_{j}')
                        y[i, t, j] = model.NewBoolVar(f'y_{i}_{t}_{j}')

        s = {t: model.NewIntVar(0, N_val * self.B, f's_{t}') for t in self.T}


        # Add all constraints
        self._add_all_constraints(model, x, y, s, M_val, N_val)

        # Create solver with enhanced heuristics
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # Optional: Print conflict statistics for debugging
        print(f"Number of conflicts: {solver.NumConflicts()}")
        print(f"Number of branches: {solver.NumBranches()}")
        print(f"Wall time: {solver.WallTime():.2f}s")

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return True, get_activity_times(solver, x)
        else:
            return False, {}

    def improved_backtracking_solver(self, M_val, N_val):
        """Enhanced backtracking with advanced heuristics and conflict learning"""
        print(f"Using Advanced Backtracking with MCV, LCV, Backjumping and No-good Learning for M={M_val}, N={N_val}")

        model = cp_model.CpModel()
        x = {}
        y = {}

        """ 
        # Create variables
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        x[i, t, j] = model.NewBoolVar(f'x_{i}_{t}_{j}')
                        y[i, t, j] = model.NewBoolVar(f'y_{i}_{t}_{j}')
        """
        
        # Create variables with tighter bounds based on domain analysis
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        # Apply node consistency - restrict domains based on unary constraints
                        can_start_at_t = (t <= self.THRESHOLD_FOR_JOB_J_AND_I.get((i, j), self.T_MAX))
                        is_silent = (i in self.silent_periods and t in self.silent_periods.get(i, []))

                        if can_start_at_t and not is_silent:
                            x[i, t, j] = model.NewBoolVar(f'x_{i}_{t}_{j}')
                            y[i, t, j] = model.NewBoolVar(f'y_{i}_{t}_{j}')
                        else:
                            # Apply node consistency - set to 0 if constraints make it impossible
                            x[i, t, j] = model.NewIntVar(0, 0, f'x_{i}_{t}_{j}')
                            y[i, t, j] = model.NewIntVar(0, 0, f'y_{i}_{t}_{j}')

        s = {t: model.NewIntVar(0, N_val * self.B, f's_{t}') for t in self.T}

        # Add all constraints
        self._add_all_constraints(model, x, y, s, M_val, N_val)
        
        # Add redundant constraints for better propagation
        self._add_redundant_constraints(model, x, y, s, M_val, N_val)

        # Create solver with enhanced heuristics
        solver = cp_model.CpSolver()

        # MOST CONSTRAINED VARIABLE: Enhanced search strategy
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH  # Let CP-SAT use its advanced MCV
        solver.parameters.cp_model_presolve = True
        solver.parameters.cp_model_probing_level = 3  # Increased for better constraint propagation
        solver.parameters.linearization_level = 2

        # BACKJUMPING: Enable conflict-driven clause learning (CDCL)
        solver.parameters.use_pb_resolution = True  # Pseudo-boolean resolution for better backjumping
        solver.parameters.clause_cleanup_period = 10000  # Manage learned clauses
        solver.parameters.clause_cleanup_target = 100000

        # NO-GOOD LEARNING: Enhanced conflict analysis
        solver.parameters.binary_minimization_algorithm = cp_model.CHOOSE_FIRST
        solver.parameters.minimize_reduction_during_pb_resolution = True

        # Advanced restart and learning strategies
        #solver.parameters.restart_algorithms = [cp_model.PORTFOLIO_WITH_QUICK_RESTART_SEARCH]
        solver.parameters.restart_period = 50
        solver.parameters.max_number_of_conflicts = 1000000

        # LEAST CONSTRAINING VALUE: Custom decision strategy
        decision_vars = []
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        decision_vars.extend([x[i, t, j], y[i, t, j]])

        # Most Constrained Variable + Least Constraining Value
        # model.AddDecisionStrategy(decision_vars,
         #                         cp_model.CHOOSE_MIN_DOMAIN_SIZE,  # MCV: choose variable with smallest domain
          #                        cp_model.SELECT_MAX_VALUE)  # LCV approximation: try max value first

        # Alternative LCV strategy - you can experiment with this instead
        model.AddDecisionStrategy(decision_vars,
                                   cp_model.CHOOSE_MIN_DOMAIN_SIZE,
                                   cp_model.SELECT_LOWER_HALF)     # Try values that are less constraining

        status = solver.Solve(model)

        # Optional: Print conflict statistics for debugging
        print(f"Number of conflicts: {solver.NumConflicts()}")
        print(f"Number of branches: {solver.NumBranches()}")
        print(f"Wall time: {solver.WallTime():.2f}s")

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return True, get_activity_times(solver, x)
        else:
            return False, {}


    def constraint_propagation_solver(self, M_val, N_val):
        """Enhanced solver with constraint propagation techniques"""
        print(f"Using Constraint Propagation for M={M_val}, N={N_val}")

        model = cp_model.CpModel()
        x = {}
        y = {}

        # Create variables with tighter bounds based on domain analysis
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in self.T:
                        # Apply node consistency - restrict domains based on unary constraints
                        can_start_at_t = (t <= self.THRESHOLD_FOR_JOB_J_AND_I.get((i, j), self.T_MAX))
                        is_silent = (i in self.silent_periods and t in self.silent_periods.get(i, []))

                        if can_start_at_t and not is_silent:
                            x[i, t, j] = model.NewBoolVar(f'x_{i}_{t}_{j}')
                            y[i, t, j] = model.NewBoolVar(f'y_{i}_{t}_{j}')
                        else:
                            # Apply node consistency - set to 0 if constraints make it impossible
                            x[i, t, j] = model.NewIntVar(0, 0, f'x_{i}_{t}_{j}')
                            y[i, t, j] = model.NewIntVar(0, 0, f'y_{i}_{t}_{j}')

        s = {t: model.NewIntVar(0, N_val * self.B, f's_{t}') for t in self.T}

        # Add constraints with enhanced propagation
        self._add_all_constraints(model, x, y, s, M_val, N_val)

        # Add redundant constraints for better propagation
        self._add_redundant_constraints(model, x, y, s, M_val, N_val)

        solver = cp_model.CpSolver()
        solver.parameters.cp_model_presolve = True
        solver.parameters.cp_model_probing_level = 3  # Enhanced probing

        status = solver.Solve(model)
        # Optional: Print conflict statistics for debugging
        print(f"Number of conflicts: {solver.NumConflicts()}")
        print(f"Number of branches: {solver.NumBranches()}")
        print(f"Wall time: {solver.WallTime():.2f}s")
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return True, get_activity_times(solver, x)
        else:
            return False, {}

    def _add_redundant_constraints(self, model, x, y, s, M_val, N_val):
        """Add redundant constraints to improve constraint propagation"""
        # Energy balance constraints
        for t in self.T:
            total_energy_consumption = sum(
                self.e[i] * x[i, t, j] for i in self.I for j in self.J if j <= self.n_jobs[i])
            model.Add(total_energy_consumption >= 0)  # Obvious but helps propagation

        # Job scheduling bounds
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    # A job can only start once
                    model.Add(sum(y[i, t, j] for t in self.T) <= 1)


    def print_solution(self, optimal_schedule):
        # Convert tuple keys to strings
        serializable_schedule = {f"{i},{j}": t for (i, j), t in optimal_schedule.items()}

        with open("optimal_schedule.json", "w") as f:
            json.dump(serializable_schedule, f, indent=2)

    def solve_with_multiple_techniques(self, M_val, N_val):
        """Try multiple techniques and return all of them"""
        techniques = [
            ("Standard Backtracking", self.standard_backtracking_solver),
            ("Constraint Propagation", self.constraint_propagation_solver),
            ("Improved Backtracking", self.improved_backtracking_solver),
        ]

        for name, technique in techniques:
            print(f"\n--- Trying {name} ---")
            try:
                start = time.time()
                el, optimal = technique(M_val, N_val)
                end = time.time()
                
                print(f"Time taken for {name}: {end - start}")
                if el:
                    print(f"️✔️ {name} found a feasible solution!")
                    self.print_solution(optimal)
                else:
                    print(f"❌ {name} did not find a feasible solution")
            except Exception as e:
                print(f"❌ {name} failed with error: {e}")

        return True


# Usage functions
def find_min_enhanced(M_val, N_val, data):
    """Enhanced version of find_min using multiple CSP techniques"""
    solver = CSPSolver(data)
    return solver.solve_with_multiple_techniques(M_val, N_val)

def print_solution_enhanced(M_val, N_val, data):
    """Enhanced solution printer that uses the best available technique"""
    solver = CSPSolver(data)

    # Try to find solution using enhanced methods
    if not solver.solve_with_multiple_techniques(M_val, N_val):
        print("No feasible solution found with any technique.")
        return

    # If solution found, use original solver to get actual variable values for visualization
    print("\nGetting detailed solution for visualization...")

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

    # Create model for solution extraction
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

    s = {t: model.NewIntVar(0, N_val * B, f's_{t}') for t in T}

    # Add all constraints
    solver._add_all_constraints(model, x, y, s, M_val, N_val)

    # Solve to get actual values
    cp_solver = cp_model.CpSolver()
    cp_solver.parameters.max_time_in_seconds = 30  # Quick solve for visualization
    status = cp_solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Visualization code
        storage_values = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        for i in I:
            print(f"\nMachine {i} Schedule:")
            for j in J:
                if j <= n_jobs[i]:
                    start_times = [t for t in T if cp_solver.Value(y[i, t, j]) > 0]
                    run_times = [t for t in T if cp_solver.Value(x[i, t, j]) > 0]
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
        ax1.set_title('Machine Schedule (Enhanced CSP Solution)')
        ax1.legend([f'Job {j}' for j in J], loc='upper right')

        for t in T:
            storage_values.append(cp_solver.Value(s[t]))

        ax2.plot(T, storage_values, marker='o', linestyle='-', markersize=4)
        battery_capacity = N_val * B
        ax2.axhline(y=battery_capacity, color='r', linestyle='--', label=f'Battery Capacity ({battery_capacity})')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Energy Storage')
        ax2.set_title('Energy Storage Levels')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('enhanced_schedule_visualization.svg', format="svg")
        print("\nEnhanced schedule visualization saved as 'enhanced_schedule_visualization.svg'")
        #plt.show()
    else:
        print("Could not extract detailed solution for visualization.")

# Enhanced solver that includes all techniques
def solve(M_val, N_val, data = get_data()):
    """Try all available CSP techniques"""
    print(f"=== Solving CSP with M={M_val}, N={N_val} using ALL techniques ===\n")

    techniques = [
        ("Enhanced CSP Solver", lambda: find_min_enhanced(M_val, N_val, data)),
    ]

    results = {}

    for name, technique in techniques:
        print(f"\n{'=' * 50}")
        print(f"TRYING: {name}")
        print(f"{'=' * 50}")

        try:
            import time
            start_time = time.time()
            success = technique()
            end_time = time.time()

            results[name] = {
                'success': success,
                'time': end_time - start_time
            }

            if success:
                print(f"{name} found a solution in {end_time - start_time:.2f} seconds")
            else:
                print(f"{name} could not find a solution ({end_time - start_time:.2f} seconds)")

        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e),
                'time': 0
            }
            print(f"ERROR in {name}: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 60}")

    successful_techniques = [name for name, result in results.items() if result['success']]

    if successful_techniques:
        print("Successful techniques:")
        for name in successful_techniques:
            print(f"   - {name} ({results[name]['time']:.2f}s)")
    else:
        print("No technique found a feasible solution")

    print(f"\n{'=' * 60}")

    return len(successful_techniques) > 0

if __name__ == "__main__":
    solve(4912, 45)
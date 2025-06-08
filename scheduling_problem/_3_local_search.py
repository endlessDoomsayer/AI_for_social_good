from ortools.linear_solver import pywraplp
import random
import matplotlib.pyplot as plt
import time
from combine_data import get_data
import _3_scip

class LocalSearchSCIPSolver:

    def __init__(self, number_of_days=7, tot_number_of_days=4011):
        self.tot_number_of_days = tot_number_of_days
        self.number_of_days = number_of_days
        self.data = get_data(number_of_days)
        self.setup_data()

    def setup_data(self):
        """Setup all the data parameters"""

        # Round to 3 decimal places
        def float_to_round(float_list):
            return {x: round(float_list[x], 3) for x in float_list}

        self.I = self.data["I"]
        self.J = self.data["J"]
        self.T = self.data["T"]
        self.n_jobs = self.data["n_jobs"]
        self.d = self.data["d"]
        self.e = float_to_round(self.data["e"])
        self.f = float_to_round(self.data["f"])
        self.c_b = self.data["c_b"]
        self.c_p = self.data["c_p"]
        self.c_e = self.data["c_e"]
        self.c_e *= self.tot_number_of_days / self.number_of_days
        self.c = self.data["c"]
        self.p = float_to_round(self.data["p"])
        self.mmm = self.data["mmm"]
        self.silent_periods = self.data["silent_periods"]
        self.M_shared = self.data["M_shared"]
        self.M_dependencies = self.data["M_dependencies"]
        self.B = self.data["B"]
        self.T_MAX = self.data["T_MAX"]
        self.THRESHOLD_FOR_JOB_J_AND_I = self.data["THRESHOLD_FOR_JOB_J_AND_I"]
        self.MACHINES = self.data["MACHINES"]

    def create_model(self, M_fixed=None, N_fixed=None):
        """Create the optimization model with optional fixed M and N values"""
        # Create the linear solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            solver = pywraplp.Solver.CreateSolver('CBC')
            if not solver:
                print('No suitable solver found.')
                return None

        # Variables
        if M_fixed is not None:
            M_var = solver.IntVar(M_fixed, M_fixed, 'M')
        else:
            M_var = solver.IntVar(0, solver.infinity(), 'M')

        if N_fixed is not None:
            N_var = solver.IntVar(N_fixed, N_fixed, 'N')
        else:
            N_var = solver.IntVar(0, solver.infinity(), 'N')

        # Binary variables for machine operations
        x = {}
        y = {}
        for i in self.I:
            for j in self.J:
                for t in self.T:
                    x[i, t, j] = solver.BoolVar(f'x_{i}_{t}_{j}')
                    y[i, t, j] = solver.BoolVar(f'y_{i}_{t}_{j}')

        # Continuous variables
        s = {t: solver.NumVar(0, solver.infinity(), f's_{t}') for t in self.T}
        z = {t: solver.NumVar(0, solver.infinity(), f'z_{t}') for t in self.T}

        # Fix variables for jobs that a machine can't do
        for i in self.I:
            for j in self.J:
                if j > self.n_jobs[i]:
                    for t in self.T:
                        x[i, t, j].SetBounds(0, 0)
                        y[i, t, j].SetBounds(0, 0)

        # Store variables in solver for later access
        solver.M_var = M_var
        solver.N_var = N_var
        solver.x = x
        solver.y = y
        solver.s = s
        solver.z = z

        # Add all constraints
        self.add_constraints(solver)

        return solver

    def add_constraints(self, solver):
        """Add all the constraints to the model"""
        M_var = solver.M_var
        N_var = solver.N_var
        x = solver.x
        y = solver.y
        s = solver.s
        z = solver.z

        # Objective: Minimize battery and power costs plus deficit
        objective = solver.Objective()
        objective.SetCoefficient(N_var, self.c_b)
        objective.SetCoefficient(M_var, self.c_p)
        for t in self.T:
            objective.SetCoefficient(z[t], self.c_e)
        objective.SetMinimization()

        # 1. Energy balance constraint
        for t in self.T:
            constraint = solver.Constraint(-solver.infinity(), 0)
            constraint.SetCoefficient(M_var, -self.p[t])
            constraint.SetCoefficient(s[t], -1)
            constraint.SetCoefficient(z[t], -1)
            for i in self.I:
                for j in self.J:
                    constraint.SetCoefficient(x[i, t, j], self.e[i])
                    constraint.SetCoefficient(y[i, t, j], self.f[i])

        # 2. Storage computation constraint
        for t in self.T:
            if t == 1:
                solver.Add(s[t] == 0)
            else:
                constraint = solver.Constraint(0, 0)
                constraint.SetCoefficient(s[t], 1)
                constraint.SetCoefficient(s[t - 1], -1)
                constraint.SetCoefficient(M_var, -self.p[t - 1])
                constraint.SetCoefficient(z[t - 1], -1)

                for i in self.I:
                    for j in self.J:
                        constraint.SetCoefficient(x[i, t - 1, j], self.e[i])
                        constraint.SetCoefficient(y[i, t - 1, j], self.f[i])

        # 3. Battery capacity constraint
        for t in self.T:
            constraint = solver.Constraint(-solver.infinity(), 0)
            constraint.SetCoefficient(s[t], 1)
            constraint.SetCoefficient(N_var, -self.B)

        # 4. Each job must run for required duration
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    constraint = solver.Constraint(self.d[i], self.d[i])
                    for t in self.T:
                        constraint.SetCoefficient(x[i, t, j], 1)

        # 5. Each machine can do one job at a time
        for i in self.I:
            for t in self.T:
                constraint = solver.Constraint(-solver.infinity(), 1)
                for j in self.J:
                    constraint.SetCoefficient(x[i, t, j], 1)

        # 6. Same for starting
        for i in self.I:
            for t in self.T:
                constraint = solver.Constraint(-solver.infinity(), 1)
                for j in self.J:
                    constraint.SetCoefficient(y[i, t, j], 1)

        # 7. Max energy constraint
        for t in self.T:
            constraint = solver.Constraint(-solver.infinity(), self.mmm[t])
            for i in self.I:
                for j in self.J:
                    constraint.SetCoefficient(x[i, t, j], self.e[i])

        # 8. Silent periods for machines
        for i in self.silent_periods:
            if i in self.I:
                for t in self.silent_periods[i]:
                    if t in self.T and t <= self.T_MAX:
                        for j in self.J:
                            constraint = solver.Constraint(0, 0)
                            constraint.SetCoefficient(x[i, t, j], 1)

        # 9. Shared resource constraint
        for t in self.T:
            for group in self.M_shared:
                machines_in_group = [i for i in group if i in self.I]
                if machines_in_group:
                    constraint = solver.Constraint(-solver.infinity(), 1)
                    for i in machines_in_group:
                        for j in self.J:
                            constraint.SetCoefficient(x[i, t, j], 1)

        # 10. Start implies run and ensure continuity constraint
        for i in self.I:
            for t in self.T:
                for j in self.J:
                    if t == 1:
                        solver.Add(x[i, t, j] == y[i, t, j])
                    else:
                        constraint = solver.Constraint(-solver.infinity(), 0)
                        constraint.SetCoefficient(x[i, t, j], 1)
                        constraint.SetCoefficient(y[i, t, j], -1)
                        constraint.SetCoefficient(x[i, t - 1, j], -1)

        # 10b. When you start a job, you must run it
        for i in self.I:
            for t in self.T:
                for j in self.J:
                    solver.Add(y[i, t, j] <= x[i, t, j])

        # 11. Dependency constraint
        for (k, kp1) in self.M_dependencies:
            if k in self.I and kp1 in self.I:
                for t in self.T:
                    for j in self.J:
                        if j <= self.n_jobs.get(k, 0) and j <= self.n_jobs.get(kp1, 0):
                            if t == 1:
                                solver.Add(y[kp1, t, j] == 0)
                            else:
                                constraint = solver.Constraint(-solver.infinity(), 0)
                                constraint.SetCoefficient(y[kp1, t, j], self.d[k])
                                for tp in range(1, t):
                                    constraint.SetCoefficient(x[k, tp, j], -1)

        # 12. Cooldown constraint
        for i in self.I:
            for t in self.T:
                if t > self.c[i]:
                    constraint = solver.Constraint(-solver.infinity(), self.n_jobs[i] * self.c[i])
                    for j in self.J:
                        constraint.SetCoefficient(y[i, t, j], self.c[i])
                        for tp in range(t - self.c[i], t):
                            constraint.SetCoefficient(x[i, tp, j], 1)

        # 13. Job must be completed before threshold
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    threshold = self.THRESHOLD_FOR_JOB_J_AND_I.get((i, j), self.T_MAX)
                    for t in range(threshold + 1, self.T_MAX + 1):
                        if t in self.T:
                            x[i, t, j].SetBounds(0, 0)

    def solve_with_timeout(self, solver, timeout=60):
        """Solve model with timeout"""
        solver.SetTimeLimit(1000 * timeout)

        try:
            status = solver.Solve()

            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                return solver.Objective().Value(), status
            else:
                return float('inf'), status
        except:
            return float('inf'), None

    def get_initial_solution(self, number_of_days, tot_number_of_days):
        """Get initial solution with high M and N values"""
        print("Finding initial solution...")

        initial_M, initial_N, obj_value = _3_scip.solve(5000,number_of_days=number_of_days,tot_number_of_days=tot_number_of_days, filename = "_local")

        solver = self.create_model(M_fixed=initial_M, N_fixed=initial_N)
        if solver is None:
            raise Exception("Could not create solver")

        if obj_value != float('inf'):
            print(f"Initial solution found: M={initial_M}, N={initial_N}, Objective={obj_value:.2f}")
            return initial_M, initial_N, obj_value, solver
        else:
            # If initial solution fails, try with really high values
            print("Initial solution failed, trying with higher values...")
            initial_M = 5000
            initial_N = 5000
            solver = self.create_model(M_fixed=initial_M, N_fixed=initial_N)
            obj_value, status = self.solve_with_timeout(solver, timeout=180)

            if obj_value != float('inf'):
                print(f"Backup initial solution found: M={initial_M}, N={initial_N}, Objective={obj_value:.2f}")
                return initial_M, initial_N, obj_value, solver
            else:
                raise Exception("Could not find initial feasible solution")

    def local_search(self, max_iterations, timeout_per_solve, number_of_days, tot_number_of_days):
        """Main local search algorithm"""
        print("Starting Local Search Optimization with SCIP...")
        start_time = time.time()

        # Get initial solution
        current_M, current_N, current_obj, current_solver = self.get_initial_solution(number_of_days=number_of_days, tot_number_of_days=tot_number_of_days)

        best_M, best_N, best_obj = current_M, current_N, current_obj
        best_solver = current_solver

        improvements = []
        no_improvement_count = 0

        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            print(f"Current solution: M={current_M}, N={current_N}, Objective={current_obj:.2f}")

            improved = False

            # Define neighborhood moves
            moves = []

            # Try reducing panels and batteries
            if current_M > 1:
                moves.extend([(-1, 0), (-2, 0), (-3, 0)])  # Reduce panels
            if current_N > 1:
                moves.extend([(0, -1), (0, -2), (0, -3)])  # Reduce batteries
            if current_M > 1 and current_N > 1:
                moves.extend([(-1, -1), (-2, -1), (-1, -2)])  # Reduce both

            # Also try small increases (in case we went too low)
            moves.extend([(1, 0), (0, 1), (1, 1)])

            # Shuffle moves to add randomness
            random.shuffle(moves)

            for delta_M, delta_N in moves:
                new_M = max(1, current_M + delta_M)
                new_N = max(1, current_N + delta_N)

                if new_M == current_M and new_N == current_N:
                    continue

                print(f"  Trying M={new_M}, N={new_N}...", end=" ")

                # Test new solution
                test_solver = self.create_model(M_fixed=new_M, N_fixed=new_N)
                if test_solver is None:
                    print("Failed to create solver")
                    continue

                test_obj, status = self.solve_with_timeout(test_solver, timeout=timeout_per_solve)

                if test_obj < current_obj:
                    print(f"IMPROVED! Objective: {test_obj:.2f}")
                    current_M, current_N, current_obj = new_M, new_N, test_obj
                    current_solver = test_solver
                    improved = True

                    if test_obj < best_obj:
                        best_M, best_N, best_obj = new_M, new_N, test_obj
                        best_solver = test_solver
                        improvements.append((iteration + 1, best_M, best_N, best_obj))

                    break  # Move to next iteration after first improvement
                else:
                    if test_obj == float('inf'):
                        print("Infeasible")
                    else:
                        print(f"No improvement: {test_obj:.2f}")

            if not improved:
                no_improvement_count += 1
                print("  No improvement found in this iteration")

                # If no improvement for several iterations, try larger jumps
                if no_improvement_count >= 3:
                    print("  Trying larger moves...")
                    large_moves = [(-5, 0), (0, -5), (-3, -3), (5, 0), (0, 5)]
                    for delta_M, delta_N in large_moves:
                        new_M = max(1, current_M + delta_M)
                        new_N = max(1, current_N + delta_N)

                        test_solver = self.create_model(M_fixed=new_M, N_fixed=new_N)
                        if test_solver is None:
                            continue

                        test_obj, status = self.solve_with_timeout(test_solver, timeout=timeout_per_solve)

                        if test_obj < current_obj:
                            current_M, current_N, current_obj = new_M, new_N, test_obj
                            current_solver = test_solver
                            improved = True
                            no_improvement_count = 0
                            break

                if no_improvement_count >= 5:
                    print("No improvement for 5 iterations. Stopping.")
                    break
            else:
                no_improvement_count = 0

        total_time = time.time() - start_time

        print(f"\n{'=' * 50}")
        print("LOCAL SEARCH COMPLETED")
        print(f"{'=' * 50}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best solution found:")
        print(f"  Number of Panels (M): {best_M}")
        print(f"  Number of Batteries (N): {best_N}")
        print(f"  Total Cost: {best_obj:.2f}")
        print(f"  Panel Cost: {best_M * self.c_p:.2f}")
        print(f"  Battery Cost: {best_N * self.c_b:.2f}")
        print(f"  Deficit Cost: {best_obj - best_M * self.c_p - best_N * self.c_b:.2f}")

        return best_M, best_N, best_obj, best_solver, improvements

    def visualize_results(self, solver, improvements):
        """Create visualizations of the results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Get solution values
        x = solver.x
        s = solver.s
        z = solver.z
        M_value = int(solver.M_var.solution_value())
        N_value = int(solver.N_var.solution_value())

        # Plot machine schedules
        for i in self.I:
            for t in self.T:
                for j in self.J:
                    if x[i, t, j].solution_value() > 0.5:
                        ax1.bar(t, 1, bottom=i - 1, color=f'C{j}', edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Machine')
        ax1.set_yticks(range(0, self.MACHINES))
        ax1.set_yticklabels([f'Machine {i}' for i in self.I])
        ax1.set_title('Machine Schedule')

        # Plot energy storage
        storage_values = [s[t].solution_value() for t in self.T]
        ax2.plot(self.T, storage_values, marker='o', linestyle='-', markersize=4)
        battery_capacity = N_value * self.B
        ax2.axhline(y=battery_capacity, color='r', linestyle='--',
                    label=f'Battery Capacity ({battery_capacity})')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Energy Storage')
        ax2.set_title('Energy Storage Levels')
        ax2.legend()

        # Plot deficit
        deficit_values = [z[t].solution_value() for t in self.T]
        ax3.plot(self.T, deficit_values, marker='s', linestyle='-', markersize=4, color='red')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Energy Deficit')
        ax3.set_title('Energy Deficit (z_t)')

        # Plot improvement history
        if improvements:
            iterations, Ms, Ns, objs = zip(*improvements)
            ax4.plot(iterations, objs, marker='o', linestyle='-', markersize=6)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Objective Value')
            ax4.set_title('Local Search Progress')
            ax4.grid(True)

        plt.tight_layout()
        plt.savefig('output/local_search_3_scip_results.png', format="png", dpi=300, bbox_inches='tight')
        print("Results visualization saved as 'output/local_search_3_scip_results.png'")


def solve(number_of_days=1, tot_number_of_days=3837):
    solver = LocalSearchSCIPSolver(number_of_days=number_of_days, tot_number_of_days=tot_number_of_days)

    # Run local search
    best_M, best_N, best_obj, best_solver, improvements = solver.local_search(
        max_iterations=1000,
        timeout_per_solve=45,
        number_of_days=number_of_days,
        tot_number_of_days=tot_number_of_days
    )

    # Visualize results
    solver.visualize_results(best_solver, improvements)
    return best_obj, best_M, best_N


if __name__ == "__main__":
    solve()
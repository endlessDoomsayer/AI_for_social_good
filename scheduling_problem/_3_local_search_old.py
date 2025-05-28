import pyomo.environ as pyo
import random
import matplotlib.pyplot as plt
import time
import copy
from combine_data import get_data
import _3_milp_old

class LocalSearchSolver:
    def __init__(self):
        # Get data
        self.data = get_data()
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
        self.c_e *= 5791/7
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
        self.BIG_M = 1000000

    def create_model(self, M_fixed=None, N_fixed=None):
        """Create the optimization model with optional fixed M and N values"""
        model = pyo.ConcreteModel()

        # Sets
        model.T = pyo.Set(initialize=self.T)
        model.I = pyo.Set(initialize=self.I)
        model.J = pyo.Set(initialize=self.J)

        # Variables
        if M_fixed is not None:
            model.M = pyo.Param(initialize=M_fixed)
        else:
            model.M = pyo.Var(domain=pyo.NonNegativeIntegers)
            
        if N_fixed is not None:
            model.N = pyo.Param(initialize=N_fixed)
        else:
            model.N = pyo.Var(domain=pyo.NonNegativeIntegers)

        model.x = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)
        model.y = pyo.Var(model.I, model.T, model.J, domain=pyo.Binary)
        model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.z = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.V = pyo.Var(model.T, domain=pyo.Reals)
        model.b = pyo.Var(model.T, domain=pyo.Binary)

        # Fix variables for jobs that a machine can't do
        for i in self.I:
            for j in self.J:
                if j > self.n_jobs[i]:
                    for t in self.T:
                        model.x[i, t, j].fix(0)
                        model.y[i, t, j].fix(0)

        # Add all your original constraints here
        self.add_constraints(model)
        
        return model

    def add_constraints(self, model):
        """Add all the constraints to the model"""
        # Objective
        def objective_rule(m):
            return m.N * self.c_b + m.M * self.c_p + self.c_e * sum(m.z[t] for t in m.T)
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Energy balance constraint
        def energy_balance_rule(m, t):
            return sum(self.e[i] * m.x[i, t, j] + self.f[i] * m.y[i, t, j] 
                      for i in m.I for j in m.J) - m.M * self.p[t] - m.s[t] <= m.z[t]
        model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

        # Volume calculation constraint
        def volume_rule(m, t):
            return m.V[t] == sum(m.M * self.p[tp] - sum(self.e[i] * m.x[i, tp, j] + self.f[i] * m.y[i, tp, j] 
                                                       for i in m.I for j in m.J)
                               for tp in range(1, t))
        model.volume_constraint = pyo.Constraint(model.T, rule=volume_rule)

        # Binary variable constraints for V_t sign
        def V_positive_rule(m, t):
            return m.V[t] >= -self.BIG_M * (1 - m.b[t])
        model.V_positive = pyo.Constraint(model.T, rule=V_positive_rule)

        def V_negative_rule(m, t):
            return m.V[t] <= self.BIG_M * m.b[t]
        model.V_negative = pyo.Constraint(model.T, rule=V_negative_rule)

        # Storage constraints
        def storage_V_rule(m, t):
            return m.s[t] <= m.V[t] + self.BIG_M * (1 - m.b[t])
        model.storage_V = pyo.Constraint(model.T, rule=storage_V_rule)

        def storage_b_rule(m, t):
            return m.s[t] <= self.BIG_M * m.b[t]
        model.storage_b = pyo.Constraint(model.T, rule=storage_b_rule)

        # Battery capacity constraint
        def battery_capacity_rule(m, t):
            return m.s[t] <= m.N * self.B
        model.battery_constraint = pyo.Constraint(model.T, rule=battery_capacity_rule)

        # Job requirements
        model.job_requirements = pyo.ConstraintList()
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    model.job_requirements.add(
                        sum(model.x[i, t, j] for t in self.T) == self.d[i]
                    )

        # Machine constraints
        def one_job_at_time(m, i, t):
            return sum(m.x[i, t, j] for j in m.J) <= 1
        model.single_job_time = pyo.Constraint(model.I, model.T, rule=one_job_at_time)

        def one_start_at_time(m, i, t):
            return sum(m.y[i, t, j] for j in m.J) <= 1
        model.single_start_time = pyo.Constraint(model.I, model.T, rule=one_start_at_time)

        # Max energy constraint
        def max_energy(m, t):
            return sum(self.e[i] * m.x[i, t, j] for i in m.I for j in m.J) <= self.mmm[t]
        model.max_energy = pyo.Constraint(model.T, rule=max_energy)

        # Silent periods
        model.silent_periods = pyo.ConstraintList()
        for i, times in self.silent_periods.items():
            for t in times:
                if t <= self.T_MAX:
                    model.silent_periods.add(sum(model.x[i, t, j] for j in self.J) == 0)

        # Shared resources
        model.shared_resources = pyo.ConstraintList()
        for t in self.T:
            for group in self.M_shared:
                machines_in_group = [i for i in group if i in self.I]
                if machines_in_group:
                    model.shared_resources.add(
                        sum(model.x[i, t, j] for i in machines_in_group for j in self.J) <= 1
                    )

        # Start-run relationships
        def run_start_relation(m, i, t, j):
            if t == 1:
                return m.x[i, t, j] == m.y[i, t, j]
            else:
                return m.x[i, t, j] <= m.y[i, t, j] + m.x[i, t - 1, j]
        model.run_start_relation = pyo.Constraint(model.I, model.T, model.J, rule=run_start_relation)

        def start_implies_run(m, i, t, j):
            return m.y[i, t, j] <= m.x[i, t, j]
        model.start_implies_run = pyo.Constraint(model.I, model.T, model.J, rule=start_implies_run)

        # Dependencies
        model.dependencies = pyo.ConstraintList()
        for (k, kp1) in self.M_dependencies:
            if k in self.I and kp1 in self.I:
                for t in self.T:
                    for j in self.J:
                        if t == 1:
                            model.dependencies.add(model.y[kp1, t, j] == 0)
                        else:
                            prev_completions = sum(model.x[k, tp, j] for tp in range(1, t))
                            model.dependencies.add(model.y[kp1, t, j] <= prev_completions / self.d[k])

        # Job completion thresholds
        model.job_completion = pyo.ConstraintList()
        for i in self.I:
            for j in self.J:
                if j <= self.n_jobs[i]:
                    for t in range(self.THRESHOLD_FOR_JOB_J_AND_I[(i, j)] + 1, self.T_MAX + 1):
                        model.job_completion.add(model.x[i, t, j] == 0)

    def solve_with_timeout(self, model, timeout=60):
        """Solve model with timeout"""
        solver = pyo.SolverFactory('glpk')
        solver.options['tmlim'] = timeout
        try:
            result = solver.solve(model, tee=False)
            if (result.solver.termination_condition == pyo.TerminationCondition.optimal or 
                result.solver.termination_condition == pyo.TerminationCondition.feasible):
                return pyo.value(model.objective), result
            else:
                return float('inf'), result
        except:
            return float('inf'), None

    def get_initial_solution(self):
        """Get initial solution with high M and N values"""
        print("Finding initial solution...")
        
        (initial_M, initial_N, obj_value) = _3_milp_old.solve(100)
        
        model = self.create_model(M_fixed=initial_M, N_fixed=initial_N)
        
        if obj_value != float('inf'):
            print(f"Initial solution found: M={initial_M}, N={initial_N}, Objective={obj_value:.2f}")
            return initial_M, initial_N, obj_value, model
        else:
            # If initial solution fails, try with even higher values
            print("Initial solution failed, trying with higher values...")
            initial_M = 12442
            initial_N = 43
            model = self.create_model(M_fixed=initial_M, N_fixed=initial_N)
            obj_value, result = self.solve_with_timeout(model, timeout=180)
            
            if obj_value != float('inf'):
                print(f"Backup initial solution found: M={initial_M}, N={initial_N}, Objective={obj_value:.2f}")
                return initial_M, initial_N, obj_value, model
            else:
                raise Exception("Could not find initial feasible solution")

    def local_search(self, max_iterations=50, timeout_per_solve=30):
        """Main local search algorithm"""
        print("Starting Local Search Optimization...")
        start_time = time.time()
        
        # Get initial solution
        current_M, current_N, current_obj, current_model = self.get_initial_solution()
        
        best_M, best_N, best_obj = current_M, current_N, current_obj
        best_model = current_model
        
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
                test_model = self.create_model(M_fixed=new_M, N_fixed=new_N)
                test_obj, result = self.solve_with_timeout(test_model, timeout=timeout_per_solve)
                
                if test_obj < current_obj:
                    print(f"IMPROVED! Objective: {test_obj:.2f}")
                    current_M, current_N, current_obj = new_M, new_N, test_obj
                    current_model = test_model
                    improved = True
                    
                    if test_obj < best_obj:
                        best_M, best_N, best_obj = new_M, new_N, test_obj
                        best_model = test_model
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
                        
                        test_model = self.create_model(M_fixed=new_M, N_fixed=new_N)
                        test_obj, result = self.solve_with_timeout(test_model, timeout=timeout_per_solve)
                        
                        if test_obj < current_obj:
                            current_M, current_N, current_obj = new_M, new_N, test_obj
                            current_model = test_model
                            improved = True
                            no_improvement_count = 0
                            break
                
                if no_improvement_count >= 5:
                    print("No improvement for 5 iterations. Stopping.")
                    break
            else:
                no_improvement_count = 0
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*50}")
        print("LOCAL SEARCH COMPLETED")
        print(f"{'='*50}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best solution found:")
        print(f"  Number of Panels (M): {best_M}")
        print(f"  Number of Batteries (N): {best_N}")
        print(f"  Total Cost: {best_obj:.2f}")
        print(f"  Panel Cost: {best_M * self.c_p:.2f}")
        print(f"  Battery Cost: {best_N * self.c_b:.2f}")
        print(f"  Deficit Cost: {best_obj - best_M * self.c_p - best_N * self.c_b:.2f}")
        
        return best_M, best_N, best_obj, best_model, improvements

    def visualize_results(self, model, improvements):
        """Create visualizations of the results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot machine schedules
        for i in self.I:
            for t in self.T:
                for j in self.J:
                    if pyo.value(model.x[i, t, j]) > 0.5:
                        ax1.bar(t, 1, bottom=i - 1, color=f'C{j}', edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Machine')
        ax1.set_yticks(range(0, self.MACHINES))
        ax1.set_yticklabels([f'Machine {i}' for i in self.I])
        ax1.set_title('Machine Schedule')
        
        # Plot energy storage
        storage_values = [pyo.value(model.s[t]) for t in self.T]
        ax2.plot(self.T, storage_values, marker='o', linestyle='-', markersize=4)
        battery_capacity = int(pyo.value(model.N)) * self.B
        ax2.axhline(y=battery_capacity, color='r', linestyle='--', 
                   label=f'Battery Capacity ({battery_capacity})')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Energy Storage')
        ax2.set_title('Energy Storage Levels')
        ax2.legend()
        
        # Plot deficit
        deficit_values = [pyo.value(model.z[t]) for t in self.T]
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
        plt.savefig('local_search_results_combined.svg', format="svg", dpi=300, bbox_inches='tight')
        #print("Results visualization saved as 'local_search_results.png'")
        plt.show()


# Usage example
def solve():
    solver = LocalSearchSolver()
    
    # Run local search
    best_M, best_N, best_obj, best_model, improvements = solver.local_search(
        max_iterations=1000, 
        timeout_per_solve=45
    )
    
    # Visualize results
    solver.visualize_results(best_model, improvements)
    
if __name__ == "__main__":
    solve()
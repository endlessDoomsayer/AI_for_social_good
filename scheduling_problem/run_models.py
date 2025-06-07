import _1_bin_search_scip
import _1_glpk
import _2
import _3_local_search
import _3_milp
import _4_1_CSP_alternatives
import _4_1_scip
import _4_3_milp
from combine_data import get_data
import pandas as pd
import time
import sys


def run_models_1(policies, days=7, date = "2018-01-01"):
    

    output_file = f"results_model_1.txt"
    number_of_M_N_per_policy = {}  

    with open(output_file, "w") as f:
        header = f"----------------------------------- MODEL 1 -----------------------------------\n"
        print(header)
        f.write(header)

        for policy in policies:

            print_policy = f"\n-----------------------------------{policy}-----------------------------------\n"
            print(print_policy)
            f.write(print_policy)

            day = pd.to_datetime(date)
            data = get_data(number_of_days=days, day=day)

            start = time.time()
            if policy == '_1_bin_search_scip':
                M, N = _1_bin_search_scip.solve(data=data)
                if M is None or N is None:
                    warning = f"No feasible (M, N) found for policy '{policy}' and date '{date}'.\n"
                    print(warning)
                    f.write(warning)
                    continue

            elif policy == '_1_glpk':
                M, N = _1_glpk.solve(data=data)
                if M is None or N is None:
                    warning = f"No feasible (M, N) found for policy '{policy}' and date '{date}'.\n"
                    print(warning)
                    f.write(warning)
                    continue
            else:
                warning = f"Policy '{policy}' not correct.\n"
                print(warning)
                f.write(warning)
                continue

            end = time.time()
            elapsed = f"Time taken for {policy}: {end - start:.2f} seconds\n"
            result = f"Results for {policy}: M={M}, N={N}\n"

            print(elapsed)
            print(result)
            f.write(f"start date: " + date + "\n")
            f.write(f"Days: {days}\n")
            f.write(f"elapsed: {elapsed}\n")
            f.write(f"result: {result}\n")

            if policy not in number_of_M_N_per_policy:
                number_of_M_N_per_policy[policy] = (M, N)


        print(f"\n----------------------------------- MODEL 1 - Summary -----------------------------------\n")
        f.write(f"\n----------------------------------- MODEL 1 - Summary -----------------------------------\n")
        for policy, (M, N) in number_of_M_N_per_policy.items():
            print(f"{policy}: M={M}, N={N}")
            f.write(f"{policy}: M={M}, N={N}\n")

    print(f"\noutput_file_saved '{output_file}'")
    return number_of_M_N_per_policy



def run_step_2(number_of_M_N, policies, number_of_days=7, date = "2018-01-01"):

    output_file = f"results_step_2.txt"
    number_of_days_years_per_policy = {}

    with open(output_file, "w") as f:
        header = f"----------------------------------- STEP 2 -----------------------------------\n"
        print(header)
        f.write(header)

        for policy in policies:

            print_policy = f"\n-----------------------------------{policy}-----------------------------------\n"
            print(print_policy)
            f.write(print_policy)

            day = pd.to_datetime(date)
            data = get_data(number_of_days=number_of_days, day=day)
 
            M, N = number_of_M_N[policy]
            days, years = _2.print_sol(M, N, number_of_days, data=data)

            result = f"Results for {policy}: Days={days}, year={years}\n"
            print(result)
            f.write(result)

            if policy not in number_of_days_years_per_policy:
                    number_of_days_years_per_policy[policy] = (days, years)

        print(f"\noutput_file_saved '{output_file}'")
        return number_of_days_years_per_policy
    


def run_models_3(policies, tot_number_of_days,  days=7, date = "2018-01-01"):
    

    output_file = f"results_model_3.txt"
    number_of_cost_M_N_per_policy = {}  

    with open(output_file, "w") as f:
        header = f"----------------------------------- MODEL 3 -----------------------------------\n"
        print(header)
        f.write(header)

        for policy in policies:

            print_policy = f"\n-----------------------------------{policy}-----------------------------------\n"
            print(print_policy)
            f.write(print_policy)

            day = pd.to_datetime(date)
            data = get_data(days, day=day)

            start = time.time()
            if policy == '_3_local_search':
                cost, M, N = _3_local_search.solve(tot_number_of_days = tot_number_of_days)
                if M is None or N is None:
                    warning = f"No feasible (M, N) found for policy '{policy}' and date '{date}'.\n"
                    print(warning)
                    f.write(warning)
                    continue
            elif policy == '_3_milp':
                cost, M, N = _3_milp.solve(tot_number_of_days = tot_number_of_days, data=data)
                if cost or M or N is None:
                    warning = f"No feasible (M, N) found for policy '{policy}' and date '{date}'.\n"
                    print(warning)
                    f.write(warning)
                    continue
            else:
                warning = f"Policy '{policy}' not correct.\n"
                print(warning)
                f.write(warning)
                continue

            end = time.time()
            elapsed = f"Time taken for {policy}: {end - start:.2f} seconds\n"
            result = f"Results for {policy}: cost={cost}, M={M}, N={N}\n"

            print(elapsed)
            print(result)
            f.write(f"start date: " + date + "\n")
            f.write(f"Days: {days}\n")
            f.write(f"Elapsed: {elapsed}\n")
            f.write(f"Result: {result}\n")

            if policy not in number_of_cost_M_N_per_policy:
                number_of_cost_M_N_per_policy[policy] = (cost, M, N)

        print(f"\n----------------------------------- MODEL 3 - Summary -----------------------------------\n")
        f.write(f"\n--------------------days--------------- MODEL 3 - Summary -----------------------------------\n")
        for policy, (M, N) in number_of_cost_M_N_per_policy.items():
            print(f"{policy}: cost={cost}, M={M}, N={N}")
            f.write(f"{policy}: cost={cost}, M={M}, N={N}\n")

    print(f"\noutput_file_saved '{output_file}'")
    return number_of_cost_M_N_per_policy


def run_models_4(number_of_M_N_per_policy, days=1, date = "2018-01-01"):

    for policy, (M, N) in number_of_M_N_per_policy.items():

        original_stdout = sys.stdout
        with open(f"results_{policy}.txt", "w") as f:
            
            sys.stdout = f  
            print_policy = f"\n-----------------------------------{policy}-----------------------------------\n"
            print(print_policy)

            day = pd.to_datetime(date)
            data = get_data(number_of_days=days, day=day)
            
            if policy == '_4_1_enhanced':
                _4_1_CSP_alternatives.solve(M, N, data = data)
            elif policy == '_4_1_lin_prog':
                _4_1_lin_prog.solve(M, N, data = data)
            elif policy == '_4_3_milp':
                _4_3_milp.solve(M, N, data = data)
            else:
                warning = f"Policy '{policy}' not correct.\n"
                print(warning)

        sys.stdout = original_stdout
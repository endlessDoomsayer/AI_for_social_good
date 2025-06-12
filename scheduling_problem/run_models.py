import _1_bin_search_scip
import _1_bin_search_scip_inverted
import _1_scip
import _2
import _3_local_search
import _3_scip
import _4_1_CSP_alternatives
import _4_1_scip
import _4_1_adapt
import _4_3_scip

from combine_data import get_data
import pandas as pd
import time
import sys

output_folder = "output/"

def run_models_1(policies, days=7, date = "2018-01-01"):

    output_file = f"results_phase_1.txt"
    number_of_M_N_per_policy = {}  

    with open(output_folder+output_file, "w") as f:
        header = f"----------------------------------- MODEL 1 -----------------------------------\n"
        print(header)
        f.write(header)

        day = pd.to_datetime(date)
        data = get_data(number_of_days=days, day=day)

        for policy in policies:

            print_policy = f"\n-----------------------------------{policy}-----------------------------------\n"
            print(print_policy)
            f.write(print_policy)

            start = time.time()
            if policy == '_1_bin_search_scip':
                M, N = _1_bin_search_scip.solve(data=data)
                if M is None or N is None:
                    warning = f"No feasible (M, N) found for policy '{policy}' and date '{date}'.\n"
                    print(warning)
                    f.write(warning)
                    continue

            elif policy == '_1_scip':
                M, N, _ = _1_scip.solve(data=data)
                if M is None or N is None:
                    warning = f"No feasible (M, N) found for policy '{policy}' and date '{date}'.\n"
                    print(warning)
                    f.write(warning)
                    continue

            elif policy == '_1_bin_search_scip_inverted':
                M, N = _1_bin_search_scip_inverted.solve(data=data)
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

    print(f"\noutput_file_saved '{output_folder+output_file}'")
    return number_of_M_N_per_policy



def run_step_2(number_of_M_N, policies, days=7, date = "2018-01-01"):

    output_file = f"results_phase_2.txt"
    number_of_days_years_per_policy = {}

    with open(output_folder+output_file, "w") as f:
        header = f"----------------------------------- STEP 2 -----------------------------------\n"
        print(header)
        f.write(header)
        day = pd.to_datetime(date)
        data = get_data(number_of_days=days, day=day)

        for policy in policies:

            print_policy = f"\n-----------------------------------{policy}-----------------------------------\n"
            print(print_policy)
            f.write(print_policy)


            M, N = number_of_M_N[policy]
            daysss, years = _2.print_sol(M, N, days, data=data)

            result = f"Results for {policy} with M={M} and N={N}: Days={daysss}, year={years}\n"
            print(result)
            f.write(result)

            if policy not in number_of_days_years_per_policy:
                    number_of_days_years_per_policy[policy] = (daysss, years)

        print(f"\noutput_file_saved '{output_folder+output_file}'")
        return number_of_days_years_per_policy
    


def run_models_3(policies, tot_number_of_days,  days=7, date = "2018-01-01"):

    output_file = f"results_phase_3.txt"
    number_of_cost_M_N_per_policy = {}  

    with open(output_folder+output_file, "w") as f:
        header = f"----------------------------------- MODEL 3 -----------------------------------\n"
        print(header)
        f.write(header)
        day = pd.to_datetime(date)
        data = get_data(days, day=day)

        for policy in policies:

            print_policy = f"\n-----------------------------------{policy}-----------------------------------\n"
            print(print_policy)
            f.write(print_policy)


            start = time.time()
            if policy == '_3_local_search':
                cost, M, N = _3_local_search.solve(tot_number_of_days = tot_number_of_days)
                if M is None or N is None or cost is None:
                    warning = f"No feasible (M, N) found for policy '{policy}' and date '{date}'.\n"
                    print(warning)
                    f.write(warning)
                    continue
            elif policy == '_3_scip':
                M, N, cost, _ = _3_scip.solve(tot_number_of_days = tot_number_of_days, data=data)
                if cost is None or M is None or N is None:
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
        f.write(f"\n----------------------------------- MODEL 3 - Summary -----------------------------------\n")
        for policy in policies:
            cost, M, N = number_of_cost_M_N_per_policy[policy]
            print(f"{policy}: cost={cost}, M={M}, N={N}")
            f.write(f"{policy}: cost={cost}, M={M}, N={N}\n")

    print(f"\noutput_file_saved '{output_folder+output_file}'")
    return number_of_cost_M_N_per_policy


def run_models_4(number_of_M_N_per_policy, days=1, date = "2018-01-01"):

    for policy, (M, N) in number_of_M_N_per_policy.items():

        print(f"Trying {policy} with M={M} and N={N}")

        original_stdout = sys.stdout
        with open(f"{output_folder}results_phase{policy}.txt", "w") as f:
            
            sys.stdout = f  
            print_policy = f"\n-----------------------------------{policy}-----------------------------------\n"
            print(print_policy)

            day = pd.to_datetime(date)
            data = get_data(number_of_days=days, day=day)
            
            if policy == '_4_1_enhanced':
                _4_1_CSP_alternatives.solve(M, N, data = data)
            elif policy == '_4_1_adapt':
                _4_1_adapt.solve(M, N, data = data)
            elif policy == '_4_1_scip':
                _4_1_scip.solve(M, N, data = data)
            elif policy == '_4_3_scip':
                _4_3_scip.solve(M, N, data = data)
            else:
                warning = f"Policy '{policy}' not correct.\n"
                print(warning)

        sys.stdout = original_stdout
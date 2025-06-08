from combine_data import get_data


def print_sol(M,N, number_of_days = 1, data = get_data()):
    print("M:", M)
    print("N:", N)

    I = data["I"]
    n_jobs = data["n_jobs"]
    d = data["d"]
    e = data["e"]
    f = data["f"]
    c_b = data["c_b"]
    c_p = data["c_p"]
    c_e = data["c_e"]

    # Calculations

    total_cost_panels = N*c_b + M*c_p

    total_energy = sum(e[i]*d[i]*n_jobs[i]+f[i]*n_jobs[i] for i in I)/number_of_days

    total_cost = (c_e*total_energy)


    print("Total cost if we buy panels:", total_cost_panels)
    print("Total cost if we don't buy panels:", total_cost)

    # Total days we could go on without panels
    days = total_cost_panels/total_cost
    print("Days if we don't buy panels:", days)
    print("Years if we don't buy panels:", days/365)

    return days, days/365

if __name__ == "__main__":
    print_sol(2491,166)
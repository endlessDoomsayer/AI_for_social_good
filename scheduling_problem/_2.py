from combine_data import get_data


def print_sol(M,N, number_of_days = 7, data = get_data()):
    print("M:", M)
    print("N:", N)

    I = data["I"]
    J = data["J"]
    T = data["T"]
    n_jobs = data["n_jobs"]
    d = data["d"]
    e = data["e"]
    f = data["f"]
    c_b = data["c_b"]
    c_p = data["c_p"]
    B = data["B"]
    c_e = data["c_e"]

    # Calculations

    total_cost_panels = N*c_b + M*c_p

    total_energy = sum(e[i]*d[i]*n_jobs[i]+f[i]*n_jobs[i] for i in I)/number_of_days

    total_cost = (c_e*total_energy)

    # using Inverter SMA Sunny Boy SB7000TL-US-22 7000 W 
    # i need to buy 2 an inverter every 24 panels 
    total_inverter = M/24

    total_cost_panels += total_inverter * 3000  # i dont found the correct price
    
    print("Total cost if we buy panels:", total_cost_panels)
    print("Total cost if we don't buy panels:", total_cost)

    # Total days we could go on without panels
    days = total_cost_panels/total_cost
    print("Days if we don't buy panels:", days)
    print("Years if we don't buy panels:", days/365)

    return days, days/365

if __name__ == "__main__":
    print_sol(4912,45)
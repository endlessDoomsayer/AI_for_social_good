from combine_data import get_data


def print_sol(M,N):
    print("M:", M)
    print("N:", N)

    # Get data
    data = get_data()

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

    total_energy = sum(e[i]*d[i]*n_jobs[i]+f[i]*n_jobs[i] for i in I)

    total_cost = (c_e*total_energy)

    # using Inverter SMA Sunny Boy SB7000TL-US-22 7000 W 
    # i need to buy 2 an inverter every 24 panels 
    total_inverter = M/24

    total_cost += total_inverter * 1200  # i dont found the correct price

    # Total days we could go on without panels
    days = total_cost_panels/total_cost
    print("Days if we don't buy panels:", days)
    print("Years if we don't buy panels:", days/365)

    return days, days/365

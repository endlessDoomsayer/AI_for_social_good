import pandas as pd

import _1_scip
import _2
import _3_scip
import combine_data

days = 1
dates = ["2018-02-19","2018-02-20","2018-02-21","2018-02-22","2018-02-23"]

# Run Model 1 for these 7 days
s_t_1 = 0
s_t_3 = 0
for date in dates:
    data = combine_data.get_data(days,pd.Timestamp(date))
    M,N,s_t_1 = _1_scip.solve(data, s_t_1,filename=date)
    print("Phase 1:",M,N)
    dayss, years = _2.print_sol(M,N,1,data)
    print("Phase 2:",dayss,years)
    M_f,N_f,_,s_t_3 = _3_scip.solve(number_of_days=days,data=data,tot_number_of_days=dayss,s_init=s_t_3,filename=date)
    print("Phase 3",M_f,N_f)
    print("Corresponds to ",_2.print_sol(M_f,N_f,1,data=data))
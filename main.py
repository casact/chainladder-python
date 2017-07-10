import chainladder as cl

RAA = cl.load_dataset('RAA')
ABC = cl.load_dataset('ABC')
UKMotor = cl.load_dataset('UKMotor')
GenIns = cl.load_dataset('GenIns')

RAA_mack = cl.MackChainladder(cl.Triangle(RAA))
ABC_mack = cl.MackChainladder(cl.Triangle(ABC))
UKMotor_mack = cl.MackChainladder(cl.Triangle(UKMotor))
GenIns_mack = cl.MackChainladder(cl.Triangle(GenIns), alpha=2, tail=True)

#print(RAA_mack.summary().round(3))
#print(ABC_mack.summary().round(3))
#cl.MackChainladder(M3IR5, tail=True)


MCL_inc = cl.load_dataset('MCLincurred')
MCL_paid = cl.load_dataset('MCLpaid')

MCL = cl.MunichChainladder(MCL_paid, MCL_inc)
BS = cl.BootChainladder(cl.Triangle(RAA),n_sims=1000, process_distr="od poisson")
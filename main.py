import chainladder as cl
import chainladder.deterministic as cld
import chainladder.stochastic as clst
import pandas as pd
import numpy

RAA = cl.load_dataset('RAA')
ABC = cl.load_dataset('ABC')
UKMotor = cl.load_dataset('UKMotor')
GenIns = cl.load_dataset('GenIns')

RAA_mack = clst.MackChainladder(cl.Triangle(RAA))
ABC_mack = clst.MackChainladder(cl.Triangle(ABC))
UKMotor_mack = clst.MackChainladder(cl.Triangle(UKMotor))
GenIns_mack = clst.MackChainladder(cl.Triangle(GenIns), alpha=2, tail=True)

print(RAA_mack.summary().round(3))
print(ABC_mack.summary().round(3))



MCL_inc = cl.load_dataset('MCLincurred')
MCL_paid = cl.load_dataset('MCLpaid')

MCL = clst.MunichChainladder(MCL_paid, MCL_inc)
print(MCL.summary())
BS = clst.BootChainladder(cl.Triangle(RAA),n_sims=1000, process_distr="od poisson")
print(BS.summary())

#prem = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]
#param_grid = {'born_ferg':{'LDF_average':['simple', 'volume', 'harmonic', 'regression','geometric'],'apriori':[.4,.55], 'exposure':[prem], 'triangle_pred':[cl.Triangle(test)]},
#             'cape_cod':{'LDF_average':['simple', 'volume', 'harmonic', 'regression','geometric'],'decay':[1,.9,.8,.7],'trend':[0,0.01,0.02,0.03],'exposure':[prem], 'triangle_pred':[cl.Triangle(test)]}}

 
#cld.Chainladder(RAA).grid_search(param_grid = param_grid)


  
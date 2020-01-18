"""
====================================
Using Industry Patterns on a Company
====================================

This example demonstrates how you can create development patterns at a
particular ``index`` grain and apply them to another.
"""
import chainladder as cl

clrd = cl.load_dataset('clrd')['CumPaidLoss']
clrd = clrd[clrd['LOB'] == 'wkcomp']

industry = clrd.sum()
allstate_industry_cl = cl.Chainladder().fit(industry).predict(clrd.loc['Allstate Ins Co Grp']).ultimate_
allstate_company_cl = cl.Chainladder().fit(clrd.loc['Allstate Ins Co Grp']).ultimate_
diff = (allstate_industry_cl - allstate_company_cl)

print(diff.rename('development',['Industry to Company LDF Diff']))

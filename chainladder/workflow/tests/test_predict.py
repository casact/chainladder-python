import numpy as np
import chainladder as cl

raa = cl.load_sample("RAA")
raa_1989 = raa[raa.valuation < raa.valuation_date]
raa_1990 = raa[raa.origin > "1981"]
raa_1990 = raa_1990[raa_1990.valuation <= raa_1990.valuation_date]
cl_ult = cl.Chainladder().fit(raa).ultimate_  # Chainladder Ultimate
apriori = cl_ult * 0 + (float(cl_ult.sum()) / 10)  # Mean Chainladder Ultimate
apriori_1989 = apriori[apriori.origin < "1990"]
apriori_1990 = apriori[apriori.origin > "1981"]


def test_voting_predict():
    bcl = cl.Chainladder()
    bf = cl.BornhuetterFerguson()
    cc = cl.CapeCod()

    estimators = [('bcl', bcl), ('bf', bf), ('cc', cc)]
    weights = np.array([[1, 2, 3]] * 3 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3)

    vot = cl.VotingChainladder(
            estimators=estimators,
            weights=weights
        ).fit(
            raa_1989,
            sample_weight=apriori_1989,
        )
    vot.predict(raa_1990, sample_weight=apriori_1990)

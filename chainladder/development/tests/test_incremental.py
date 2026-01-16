import chainladder as cl
import numpy as np


def test_schmidt():
    tri = cl.load_sample("ia_sample")
    xp = np if tri.array_backend == "sparse" else tri.get_array_module()
    ia = cl.IncrementalAdditive()
    ia_transform = ia.fit_transform(
        tri.iloc[0, 0], sample_weight=tri.iloc[0, 1].latest_diagonal
    )
    answer = ia_transform.incremental_.incr_to_cum().values[0, 0, :, -1]
    answer_zeta = ia_transform.zeta_.values[0, 0, 0, :]
    answer_cum_zeta = ia_transform.cum_zeta_.values[0, 0, 0, :]
    check = xp.array(
        [
            3483.0,
            4007.84795031,
            4654.36196862,
            5492.00685523,
            6198.10197128,
            7152.82539296,
        ]
    )
    check_zeta = xp.array(
        [0.24321225, 0.22196026, 0.15397836, 0.14185271, 0.09067327, 0.03677019]
    )
    check_cum_zeta = xp.array(
        [0.88844704, 0.64523479, 0.42327453, 0.26929617, 0.12744346, 0.03677019]
    )
    assert (
        xp.allclose(answer, check, atol=1e-5)
        & xp.allclose(answer_zeta, check_zeta, atol=1e-8)
        & xp.allclose(answer_cum_zeta, check_cum_zeta, atol=1e-8)
    )


def test_IBNR_methods():
    tri = cl.load_sample("ia_sample")
    incr_est = cl.IncrementalAdditive().fit(
        tri["loss"], sample_weight=tri["exposure"].latest_diagonal
    )
    incr_trans = incr_est.transform(tri["loss"])
    incr_tri = incr_est.incremental_.incr_to_cum()
    incr_ult = incr_tri[incr_tri.development == incr_tri.development.max()]
    cl_est = cl.Chainladder().fit(incr_trans)
    bf_est = cl.BornhuetterFerguson(apriori=1).fit(incr_trans, sample_weight=incr_ult)
    incr_result = np.round(incr_est.incremental_.values, 5)
    cl_result = np.round(cl_est.full_triangle_.cum_to_incr().values[..., :-2], 5)
    bf_result = np.round(bf_est.full_triangle_.cum_to_incr().values[..., :-2], 5)
    assert np.all(incr_result == cl_result) & np.all(incr_result == bf_result)


def test_pipeline():
    clrd = cl.load_sample("clrd").groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    dev = cl.Development().fit_transform(clrd)
    ult = cl.Chainladder().fit(clrd)
    dev1 = cl.IncrementalAdditive(
        n_periods=7,
        drop_valuation=1995,
        drop=("1992", 12),
        drop_above=1.05,
        drop_below=-1,
        drop_high=1,
        drop_low=1,
    ).fit(clrd, sample_weight=ult.ultimate_ * 3)
    pipe = cl.Pipeline(
        steps=[
            ("n_periods", cl.IncrementalAdditive(n_periods=7)),
            ("drop_valuation", cl.IncrementalAdditive(drop_valuation=1995)),
            ("drop", cl.IncrementalAdditive(drop=("1992", 12))),
            (
                "drop_abovebelow",
                cl.IncrementalAdditive(drop_above=1.05, drop_below=0.95),
            ),
            ("drop_hilo", cl.IncrementalAdditive(drop_high=1, drop_low=1)),
        ]
    )
    dev2 = pipe.fit(X=clrd, sample_weight=ult.ultimate_ * 3)
    assert np.array_equal(
        dev1.zeta_.values, dev2.named_steps.drop_hilo.zeta_.values, True
    )

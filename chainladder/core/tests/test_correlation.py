import chainladder as cl


def val_corr_p(data, ci):
    return cl.load_sample(data).valuation_correlation(p_critical=ci, total=True)
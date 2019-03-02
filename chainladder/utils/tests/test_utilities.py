import chainladder as cl


def test_non_vertical_line():
    true_olf = (1-.5*(184/365)**2)*.2
    olf_low = cl.parallelogram_olf([.20],['7/1/2017'], grain='Y').loc['2017'].iloc[0]-1
    olf_high = cl.parallelogram_olf([.20],['7/2/2017'], grain='Y').loc['2017'].iloc[0]-1
    assert olf_low < true_olf < olf_high


def test_vertical_line():
    olf = cl.parallelogram_olf([.20], ['7/1/2017'], grain='Y', vertical_line=True)
    assert abs(olf.loc['2017'].iloc[0] - ((1-184/365)*.2+1)) < .00001

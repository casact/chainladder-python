import chainladder as cl


def test_constant_balances(qtr):
    xp = qtr.get_array_module()
    assert (
        round(
            float(
                xp.prod(
                    cl
                    .TailConstant(1.05, decay=0.8)
                    .fit(qtr)
                    .ldf_.iloc[0, 1]
                    .values[0, 0, 0, -5:]
                )
            ),
            3,
        )
        == 1.050
    )


def test_constant_attachment_age():
    tri = cl.load_sample("genins")
    tail = cl.TailConstant(tail=1.05, attachment_age=48).fit(tri)
    assert tail.ldf_ is not None

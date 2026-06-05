"""Central registry of bundled sample datasets.

Single source of truth for the metadata of every CSV in
``chainladder/utils/data/``. Consumed by:

* :func:`chainladder.load_sample` -- to build the ``Triangle`` for a sample.
* :func:`chainladder.list_samples` -- to list available samples.
* ``docs/library/sample_data.ipynb`` -- renders the sample table live via
  ``cl.list_samples()``.
* ``MANIFEST.in`` -- ships ``chainladder/utils/data/*.csv`` via a wildcard.

Adding a new sample dataset is a one-entry change here (plus dropping the
CSV in ``chainladder/utils/data/``); ``load_sample``, the docs table, and the
tests all key off this dict, so the metadata no longer has to be repeated in
three places.

Each entry maps the sample name (the CSV filename without extension, lower
case) to the keyword arguments passed to ``Triangle``:

``origin``
    Column name(s) for the origin period.
``development``
    Column name(s) for the development period.
``index``
    Column name(s) used as the Triangle index, or ``None``.
``columns``
    Measure column name(s) loaded into the Triangle.
``cumulative``
    ``True`` if the measures are cumulative, ``False`` if incremental.
``development_format``
    Optional. Passed to ``Triangle`` by :func:`chainladder.load_sample`.
"""

SAMPLES: dict = {
    "abc": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["values"],
        "cumulative": True,
    },
    "auto": {
        "origin": "origin",
        "development": "development",
        "index": ["lob"],
        "columns": ["incurred", "paid"],
        "cumulative": True,
    },
    "berqsherm": {
        "origin": "AccidentYear",
        "development": "DevelopmentYear",
        "index": ["LOB"],
        "columns": ["Incurred", "Paid", "Reported", "Closed"],
        "cumulative": True,
    },
    "cc_sample": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["loss", "exposure"],
        "cumulative": True,
    },
    "clrd": {
        "origin": "AccidentYear",
        "development": "DevelopmentYear",
        "index": ["GRNAME", "LOB"],
        "columns": [
            "IncurLoss",
            "CumPaidLoss",
            "BulkLoss",
            "EarnedPremDIR",
            "EarnedPremCeded",
            "EarnedPremNet",
        ],
        "cumulative": True,
    },
    "clrd2025": {
        "origin": "AccidentYear",
        "development": "DevelopmentYear",
        "index": ["GRNAME", "LOB"],
        "columns": [
            "IncurredLosses",
            "CumPaidLoss",
            "BulkLoss",
            "EarnedPremDIR",
            "EarnedPremCeded",
            "EarnedPremNet",
        ],
        "cumulative": True,
    },
    "friedland_auto_bi_insurer": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Paid Claims", "Reported Claims", "Earned Premium"],
        "cumulative": True,
    },
    "friedland_auto_freq_sev": {
        "origin": "Accident Half-Year",
        "development": "Calendar Half-Year",
        "index": None,
        "columns": [
            "Closed Claim Counts",
            "Reported Claim Counts",
            "Reported Claims",
            "Reported Severity",
            "Paid Claims",
        ],
        "cumulative": True,
    },
    "friedland_auto_salsub": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": [
            "Reported Salvage and Subrogation",
            "Received Salvage and Subrogation",
            "Reported Claims",
            "Paid Claims",
        ],
        "cumulative": True,
    },
    "friedland_autoprop": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Reported ALAE", "Paid ALAE", "Reported Claims", "Paid Claims"],
        "cumulative": True,
    },
    "friedland_berq_sher_auto": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": [
            "Paid Claims",
            "Closed Claim Counts",
            "Reported Claim Counts",
            "Disposal Rate",
        ],
        "cumulative": True,
    },
    "friedland_gl_insurer": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": [
            "Closed Claim Counts",
            "Reported Claim Counts",
            "Disposal Rate",
            "Paid Claims",
        ],
        "cumulative": True,
    },
    "friedland_gl_self_insurer": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": [
            "Reported Claims",
            "Paid Claims",
        ],
        "cumulative": True,
        "development_format": "%Y-12-31",
    },
    "friedland_med_mal": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": [
            "Reported Claims",
            "Paid Claims",
            "Case Outstanding",
            "Open Claim Counts",
        ],
        "cumulative": True,
    },
    "friedland_qs": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Gross Reported Claims", "Net Reported Claims", "Net to Gross"],
        "cumulative": True,
    },
    "friedland_us_auto_chg_prod_mix": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Paid Claims", "Reported Claims"],
        "cumulative": True,
    },
    "friedland_us_auto_incr_claim": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Paid Claims", "Reported Claims"],
        "cumulative": True,
    },
    "friedland_us_auto_steady_state": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Paid Claims", "Reported Claims"],
        "cumulative": True,
    },
    "friedland_us_industry_auto": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Paid Claims", "Reported Claims", "Earned Premium"],
        "cumulative": True,
    },
    "friedland_us_industry_auto_case": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Case Outstanding", "Paid Claims"],
        "cumulative": True,
    },
    "friedland_uspp_auto_increasing_case": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Reported Claims", "Paid Claims", "Earned Premium"],
        "cumulative": True,
    },
    "friedland_uspp_auto_increasing_claim": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Reported Claims", "Paid Claims", "Earned Premium"],
        "cumulative": True,
    },
    "friedland_uspp_auto_steady_state": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Reported Claims", "Paid Claims", "Earned Premium"],
        "cumulative": True,
    },
    "friedland_uspp_increasing_claim_case": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Reported Claims", "Paid Claims", "Earned Premium"],
        "cumulative": True,
    },
    "friedland_wc_self_insurer": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": [
            "Closed Claim Counts",
            "Reported Claim Counts",
            "Paid Claims",
            "Paid Severities",
            "Reported Claims",
            "Reported Severities",
            "Payroll",
        ],
        "cumulative": True,
    },
    "friedland_xol": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": [
            "Gross Reported Claims",
            "Net Reported Claims",
            "Ceded Reported Claims",
        ],
        "cumulative": True,
    },
    "friedland_xyz_auto_bi": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Paid Claims", "Reported Claims", "Closed Claim Counts", "Reported Claim Counts", "Case Outstanding", "Reported Severities"],
        "cumulative": True,
    },
    "friedland_xyz_disp": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Disposal Rate", "Closed Claim Counts", "Paid Claims"],
        "cumulative": True,
    },
    "genins": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["values"],
        "cumulative": True,
    },
    "ia_sample": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["loss", "exposure"],
        "cumulative": True,
    },
    "liab": {
        "origin": "origin",
        "development": "development",
        "index": ["lob"],
        "columns": ["values"],
        "cumulative": True,
    },
    "m3ir5": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["values"],
        "cumulative": True,
    },
    "mack_1997": {
        "origin": "Accident Year",
        "development": "Calendar Year",
        "index": None,
        "columns": ["Case Incurred"],
        "cumulative": True,
    },
    "mcl": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["incurred", "paid"],
        "cumulative": True,
    },
    "mortgage": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["values"],
        "cumulative": True,
    },
    "mw2008": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["values"],
        "cumulative": True,
    },
    "mw2014": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["values"],
        "cumulative": True,
    },
    "prism": {
        "origin": "AccidentDate",
        "development": "PaymentDate",
        "index": ["ClaimNo", "Line", "Type", "ClaimLiability", "Limit", "Deductible"],
        "columns": ["reportedCount", "closedPaidCount", "Paid", "Incurred"],
        "cumulative": False,
    },
    "quarterly": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["incurred", "paid"],
        "cumulative": True,
    },
    "raa": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["values"],
        "cumulative": True,
    },
    "tail_sample": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["incurred", "paid"],
        "cumulative": True,
    },
    "ukmotor": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["values"],
        "cumulative": True,
    },
    "usaa": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["incurred", "paid"],
        "cumulative": True,
    },
    "usauto": {
        "origin": "origin",
        "development": "development",
        "index": None,
        "columns": ["incurred", "paid"],
        "cumulative": True,
    },
    "xyz": {
        "origin": "AccidentYear",
        "development": "DevelopmentYear",
        "index": None,
        "columns": ["Incurred", "Paid", "Reported", "Closed", "Premium"],
        "cumulative": True,
    },
}

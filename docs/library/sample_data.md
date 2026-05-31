# Sample Dataset

Below is the list of all datasets that come included with the `chainladder` package, and their basic attributes.

You can load any dataset with `cl.load_sample(...)` such as `cl.load_sample("abc")`.

This table is generated from the sample-dataset manifest
(`chainladder/utils/data/_manifest.py`) via `cl.list_samples()`. To regenerate it,
run `python scripts/regen_sample_data_docs.py` from the repository root.

| Dataset Name | Indexes | Columns | Origin Grain | Development Grain |
|---|---|---|---|---|
| abc | (none) | [values] | Annual (11 Yrs) | Annual (11 Yrs) |
| auto | [lob] | [incurred, paid] | Annual (10 Yrs) | Annual (10 Yrs) |
| berqsherm | [LOB] | [Incurred, Paid, Reported, Closed] | Annual (8 Yrs) | Annual (8 Yrs) |
| cc_sample | (none) | [loss, exposure] | Annual (5 Yrs) | Annual (5 Yrs) |
| clrd | [GRNAME, LOB] | [IncurLoss, CumPaidLoss, BulkLoss, EarnedPremDIR, EarnedPremCeded, EarnedPremNet] | Annual (10 Yrs) | Annual (10 Yrs) |
| clrd2025 | [GRNAME, LOB] | [IncurredLosses, CumPaidLoss, BulkLoss, EarnedPremDIR, EarnedPremCeded, EarnedPremNet] | Annual (19 Yrs) | Annual (19 Yrs) |
| friedland_auto_bi_insurer | (none) | [Paid Claims, Reported Claims] | Annual (9 Yrs) | Annual (9 Yrs) |
| friedland_auto_freq_sev | (none) | [Closed Claim Counts, Reported Claim Counts, Reported Claims, Reported Severity] | Semiannual (10 Half-Yrs) | Semiannual (10 Half-Yrs) |
| friedland_auto_salsub | (none) | [Reported Salvage and Subrogation, Received Salvage and Subrogation, Reported Claims, Paid Claims] | Annual (11 Yrs) | Annual (11 Yrs) |
| friedland_autoprop | (none) | [Reported ALAE, Paid ALAE, Reported Claims, Paid Claims] | Annual (11 Yrs) | Annual (11 Yrs) |
| friedland_berq_sher_auto | (none) | [Paid Claims, Closed Claim Counts, Reported Claim Counts, Disposal Rate] | Annual (8 Yrs) | Annual (8 Yrs) |
| friedland_gl_insurer | (none) | [Closed Claim Counts, Reported Claim Counts, Disposal Rate, Paid Claims] | Annual (8 Yrs) | Annual (8 Yrs) |
| friedland_med_mal | (none) | [Reported Claims, Paid Claims, Case Outstanding, Open Claim Counts] | Annual (8 Yrs) | Annual (8 Yrs) |
| friedland_qs | (none) | [Gross Reported Claims, Net Reported Claims, Net to Gross] | Annual (4 Yrs) | Annual (4 Yrs) |
| friedland_us_auto_chg_prod_mix | (none) | [Paid Claims, Reported Claims] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_us_auto_incr_claim | (none) | [Paid Claims, Reported Claims] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_us_auto_steady_state | (none) | [Paid Claims, Reported Claims] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_us_industry_auto | (none) | [Paid Claims, Reported Claims] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_us_industry_auto_case | (none) | [Case Outstanding, Paid Claims] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_uspp_auto_increasing_case | (none) | [Reported Claims, Paid Claims, Earned Premium] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_uspp_auto_increasing_claim | (none) | [Reported Claims, Paid Claims, Earned Premium] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_uspp_auto_steady_state | (none) | [Reported Claims, Paid Claims, Earned Premium] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_uspp_increasing_claim_case | (none) | [Reported Claims, Paid Claims, Earned Premium] | Annual (10 Yrs) | Annual (10 Yrs) |
| friedland_wc_self_insurer | (none) | [Closed Claim Counts, Reported Claim Counts, Paid Claims, Paid Severities, Reported Claims, Reported Severities] | Annual (8 Yrs) | Annual (8 Yrs) |
| friedland_xol | (none) | [Gross Reported Claims, Net Reported Claims, Ceded Reported Claims] | Annual (4 Yrs) | Annual (4 Yrs) |
| friedland_xyz_auto_bi | (none) | [Paid Claims, Reported Claims] | Annual (11 Yrs) | Annual (11 Yrs) |
| friedland_xyz_case | (none) | [Case Outstanding, Paid Claims] | Annual (11 Yrs) | Annual (11 Yrs) |
| friedland_xyz_disp | (none) | [Disposal Rate, Closed Claim Counts, Paid Claims] | Annual (8 Yrs) | Annual (8 Yrs) |
| friedland_xyz_freq_sev | (none) | [Closed Claim Counts, Reported Claim Counts, Reported Claims, Reported Severities] | Annual (11 Yrs) | Annual (11 Yrs) |
| genins | (none) | [values] | Annual (10 Yrs) | Annual (10 Yrs) |
| ia_sample | (none) | [loss, exposure] | Annual (6 Yrs) | Annual (6 Yrs) |
| liab | [lob] | [values] | Annual (14 Yrs) | Annual (14 Yrs) |
| m3ir5 | (none) | [values] | Annual (14 Yrs) | Annual (14 Yrs) |
| mack_1997 | (none) | [Case Incurred] | Annual (10 Yrs) | Annual (10 Yrs) |
| mcl | (none) | [incurred, paid] | Annual (7 Yrs) | Annual (7 Yrs) |
| mortgage | (none) | [values] | Annual (9 Yrs) | Annual (9 Yrs) |
| mw2008 | (none) | [values] | Annual (9 Yrs) | Annual (9 Yrs) |
| mw2014 | (none) | [values] | Annual (17 Yrs) | Annual (17 Yrs) |
| prism | [ClaimNo, Line, Type, ClaimLiability, Limit, Deductible] | [reportedCount, closedPaidCount, Paid, Incurred] | Month (120 months) | Month (120 months) |
| quarterly | (none) | [incurred, paid] | Annual (12 Yrs) | Quarter (45 Qtrs) |
| raa | (none) | [values] | Annual (10 Yrs) | Annual (10 Yrs) |
| tail_sample | (none) | [incurred, paid] | Annual (10 Yrs) | Annual (10 Yrs) |
| ukmotor | (none) | [values] | Annual (7 Yrs) | Annual (7 Yrs) |
| usaa | (none) | [incurred, paid] | Annual (10 Yrs) | Annual (10 Yrs) |
| usauto | (none) | [incurred, paid] | Annual (10 Yrs) | Annual (10 Yrs) |
| xyz | (none) | [Incurred, Paid, Reported, Closed, Premium] | Annual (11 Yrs) | Annual (11 Yrs) |

# Sample Dataset

Below is the list of all datasets that come included with the `chainladder` package, and their basic attributes.

You can load any dataset with `cl.load_sample(...)` such as `cl.load_sample("abc")`.

\* Denotes datasets that are more interesting and possess unique characteristics.


| Dataset Name | Indexes                                                  | Columns                                                                             | Origin Grain       | Development Grain  |
|--------------|----------------------------------------------------------|-------------------------------------------------------------------------------------|--------------------|--------------------|
| abc          | (none)                                                   | (none)                                                                              | Annual (11 Yrs)    | Annual (11 Yrs)    |
| auto         | [lob]                                                    | [incurred, paid]                                                                    | Annual (10 Yrs)    | Annual (10 Yrs)    |
| berqsherm    | [LOB]                                                    | [Incurred, Paid, Reported, Closed]                                                  | Annual (8 Yrs)     | Annual (8 Yrs)     |
| cc_sample    | [Total]                                                  | [loss, exposure]                                                                    | Annual (5 Yrs)     | Annual (5 Yrs)     |
| clrd *       | [GRNAME, LOB]                                            | [IncurLoss, CumPaidLoss, BulkLoss, EarnedPremDIR, EarnedPremCeded,   EarnedPremNet] | Annual (10 Yrs)    | Annual (10 Yrs)    |
| genins       | (none)                                                   | (none)                                                                              | Annual (10 Yrs)    | Annual (10 Yrs)    |
| ia_sample    | [Total]                                                  | [loss, exposure]                                                                    | Annual (6 Yrs)     | Annual (6 Yrs)     |
| liab *       | [lob]                                                    | [values]                                                                            | Annual (14 Yrs)    | Annual (14 Yrs)    |
| m3ir5        | (none)                                                   | (none)                                                                              | Annual (14 Yrs)    | Annual (14 Yrs)    |
| mcl          | [Total]                                                  | [incurred,  paid]                                                                   | Annual (7 Yrs)     | Annual (7 Yrs)     |
| mortgage     | (none)                                                   | (none)                                                                              | Annual (9 Yrs)     | Annual (9 Yrs)     |
| mw2008       | (none)                                                   | (none)                                                                              | Annual (9 Yrs)     | Annual (9 Yrs)     |
| mw2014       | (none)                                                   | (none)                                                                              | Annual (17 Yrs)    | Annual (17 Yrs)    |
| prism *      | [ClaimNo, Line, Type, ClaimLiability, Limit, Deductible] | [reportedCount, closedPaidCount, Paid, Incurred]                                    | Month (120 months) | Month (120 months) |
| quarterly *  | [Total]                                                  | [incurred,  paid]                                                                   | Annual (12 Yrs)    | Quarter (45 Qtrs)  |
| raa          | (none)                                                   | (none)                                                                              | Annual (10 Yrs)    | Annual (10 Yrs)    |
| tail_sample  | [Total]                                                  | [incurred, paid]                                                                    | Annual (10 Yrs)    | Annual (10 Yrs)    |
| ukmotor      | (none)                                                   | (none)                                                                              | Annual (7 Yrs)     | Annual (7 Yrs)     |
| usaa         | [Total]                                                  | [incurred,  paid]                                                                   | Annual (10 Yrs)    | Annual (10 Yrs)    |
| usauto       | [Total]                                                  | [incurred,  paid]                                                                   | Annual (10 Yrs)    | Annual (10 Yrs)    |
| xyz          |        [Total]                                           |        [Incurred, Paid, Reported, Closed, Premium]                                  | Annual (11 Yrs)    | Annual (11 Yrs)    |

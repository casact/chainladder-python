# Sample Dataset

Below is the list of all datasets that come included with the `chainladder` package, and their basic attributes.

You can load any dataset with `cl.load_sample(...)` such as `cl.load_sample("abc")`. * Denotes datasets that are more interesting and possess unique characteristics.


| Dataset Name | Index           | Columns                                                                             | Origin Gran     | Development Grain | Description                                            |
| ------------ | --------------- | ----------------------------------------------------------------------------------- | --------------- | ----------------- | ------------------------------------------------------ |
| abc          | *(none)*        | *(none)*                                                                            | Annual (11 Yrs) | Annual (11 Yrs)   |                                                        |
| auto         | \[lob\]         | \[incurred, paid\]                                                                  | Annual (10 Yrs) | Annual (10 Yrs)   |                                                        |
| berqsherm    | \[LOB\]         | \[Incurred, Paid, Reported, Closed\]                                                | Annual (8 Yrs)  | Annual (8 Yrs)    | Data from the Berquist Sherman paper                   |
| cc\_sample   | \[Total\]       | \[loss, exposure\]                                                                  | Annual (5 Yrs)  | Annual (5 Yrs)    | Sample insurance data for Cape Cod method in Struhuss  |
| clrd *       | \[GRNAME, LOB\] | \[IncurLoss, CumPaidLoss, BulkLoss, EarnedPremDIR, EarnedPremCeded, EarnedPremNet\] | Annual (10 Yrs) | Annual (10 Yrs)   | CAS Loss Reserving Database                            |
| genins       | *(none)*        | *(none)*                                                                            | Annual (10 Yrs) | Annual (10 Yrs)   | General insurance data used in Clark                   |
| ia\_sample   | \[Total\]       | \[loss, exposure\]                                                                  | Annual (6 Yrs)  | Annual (6 Yrs)    | Sample data for Incremental Additive Method in Schmidt |
| liab *       | \[lob\]         | \[values\]                                                                          | Annual (14 Yrs) | Annual (14 Yrs)   |                                                        |
| m3ir5        | *(none)*        | *(none)*                                                                            | Annual (14 Yrs) | Annual (14 Yrs)   |                                                        |
| mcl          | \[Total\]       | \[incurred, paid\]                                                                  | Annual (7 Yrs)  | Annual (7 Yrs)    | Sample insurance data for Munich Adjustment in Quarg   |
| mortgage     | *(none)*        | *(none)*                                                                            | Annual (9 Yrs)  | Annual (9 Yrs)    |                                                        |
| mw2008       | *(none)*        | *(none)*                                                                            | Annual (9 Yrs)  | Annual (9 Yrs)    |                                                        |
| mw2014       | *(none)*        | *(none)*                                                                            | Annual (17 Yrs) | Annual (17 Yrs)   |                                                        |
| quarterly *  | \[Total\]       | \[incurred, paid\]                                                                  | Annual (12 Yrs) | Quarter (45 Qtrs) |
| raa          | *(none)*        | *(none)*                                                                            | Annual (10 Yrs) | Annual (10 Yrs)   | Sample data used in Mack Chainladder                   |
| ukmotor      | *(none)*        | *(none)*                                                                            | Annual (7 Yrs)  | Annual (7 Yrs)    |                                                        |
| usaa         | \[Total\]       | \[incurred, paid\]                                                                  | Annual (10 Yrs) | Annual (10 Yrs)   |                                                        |
| usauto       | \[Total\]       | \[incurred, paid\]                                                                  | Annual (10 Yrs) | Annual (10 Yrs)   |                                                        |

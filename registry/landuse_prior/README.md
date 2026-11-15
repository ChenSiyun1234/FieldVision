# Land-use prior per county

`county_prior.csv`: all 2,890 counties with CDL county statistics (from the county back-test work);
`in_census2022_fit` marks the 1,410 counties with a published 2022 Census silage share. For each,
the share of hay, alfalfa, clover and vetch in its cropland (mean of the 2019-2023 Cropland Data
Layer, `forage_share_mean1923`) and the prior silage share 0.016 + 0.674 x that share, clipped to
0-1. The line was fitted by the county back-test work on the 2022 Census shares of these same
counties (leave one state out: county error 5.9 points, state error 4.2), so on any of them the
prior is not an independent check.

It is a proxy: what a usual year looks like in that landscape, nothing about the season or any field.
The toolbox shows it beside the satellite figure, with the mean of the two, and never uses it as
the headline.

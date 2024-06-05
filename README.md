# NYC Crime Rate Analysis

In this analysis we synthesize NYC crime data with census data, and zillow housing data to understand the effect of crime rates on housing prices at a zip code level. We show that there is a significant negative relationship ( p < 0.0003, R^2 = 0.311) between the trend in crime rate per 100,000 and the percent change in housing prices. On average, if the linear crime rate trend increases by 1 per 100,000 people, this causes the average NYC home to lose $5,312 in value.  Along the way we build a model to predict the profile of a criminal suspect given the profile of a victim. 

## Data Visualization - Idle hands are the Devil's workshop

### 

We start by getting NYC crime report data from NYC Open Data : CSV data https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data

We also get home values by zip code from Zillow Home Value Index (ZHVI All home SFT, Condo/Coop) Time Series Smoothed seasonally adjusted : CSV data https://www.zillow.com/research/data/

Finally we need population data by zip code : For this we look at the Census data.

https://data.census.gov/table/ACSDT5Y2011.B01003?t=Counts,%20Estimates,%20and%20Projections:Population%20Total&g=040XX00US36$8600000&y=2011

For the raw CSV files see ![data](notebook/link to data.txt)


![Test Chart](notebook/output_48.png)

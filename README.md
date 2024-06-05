# NYC Crime Rate Analysis

In this analysis we synthesize NYC crime data with census data, and zillow housing data to understand the effect of crime rates on housing prices at a zip code level. We show that there is a significant negative relationship ( p < 0.0003, R^2 = 0.311) between the trend in crime rate per 100,000 and the percent change in housing prices. On average, if the linear crime rate trend increases by 1 per 100,000 people, this causes the average NYC home to lose $5,312 in value.  Along the way we build a model to predict the profile of a criminal suspect given the profile of a victim. 

## Data Visualization

### Data

We start by getting NYC crime report data from NYC Open Data : CSV data https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data

We also get home values by zip code from Zillow Home Value Index (ZHVI All home SFT, Condo/Coop) Time Series Smoothed seasonally adjusted : CSV data https://www.zillow.com/research/data/

Finally we need population data by zip code : For this we look at the Census data.

https://data.census.gov/table/ACSDT5Y2011.B01003?t=Counts,%20Estimates,%20and%20Projections:Population%20Total&g=040XX00US36$8600000&y=2011

For the raw CSV files see: https://drive.google.com/drive/folders/19QnftB8seO4_HQYX9Umu3FkkGjwCW9PI?usp=drive_link

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CMPLNT_FR_DT</th>
      <th>LAW_CAT_CD</th>
      <th>SUSP_AGE_GROUP</th>
      <th>SUSP_RACE</th>
      <th>SUSP_SEX</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>VIC_AGE_GROUP</th>
      <th>VIC_RACE</th>
      <th>VIC_SEX</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>03/10/2008</td>
      <td>FELONY</td>
      <td>&lt;18</td>
      <td>BLACK</td>
      <td>M</td>
      <td>40.650142</td>
      <td>-73.944674</td>
      <td>25-44</td>
      <td>BLACK</td>
      <td>M</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12/21/2008</td>
      <td>MISDEMEANOR</td>
      <td>25-44</td>
      <td>BLACK</td>
      <td>M</td>
      <td>40.669126</td>
      <td>-73.973071</td>
      <td>25-44</td>
      <td>WHITE</td>
      <td>M</td>
    </tr>
    <tr>
      <th>9</th>
      <td>04/19/2008</td>
      <td>VIOLATION</td>
      <td>18-24</td>
      <td>WHITE HISPANIC</td>
      <td>M</td>
      <td>40.689954</td>
      <td>-73.916924</td>
      <td>25-44</td>
      <td>WHITE HISPANIC</td>
      <td>F</td>
    </tr>
    <tr>
      <th>20</th>
      <td>07/14/2008</td>
      <td>FELONY</td>
      <td>25-44</td>
      <td>BLACK</td>
      <td>F</td>
      <td>40.628285</td>
      <td>-73.944245</td>
      <td>45-64</td>
      <td>WHITE</td>
      <td>M</td>
    </tr>
    <tr>
      <th>26</th>
      <td>11/11/2008</td>
      <td>MISDEMEANOR</td>
      <td>25-44</td>
      <td>BLACK HISPANIC</td>
      <td>M</td>
      <td>40.655604</td>
      <td>-73.926420</td>
      <td>45-64</td>
      <td>BLACK</td>
      <td>F</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8914821</th>
      <td>08/06/2008</td>
      <td>VIOLATION</td>
      <td>65+</td>
      <td>BLACK</td>
      <td>F</td>
      <td>40.834776</td>
      <td>-73.916017</td>
      <td>45-64</td>
      <td>BLACK</td>
      <td>F</td>
    </tr>
    <tr>
      <th>8914824</th>
      <td>10/27/2008</td>
      <td>MISDEMEANOR</td>
      <td>25-44</td>
      <td>BLACK</td>
      <td>M</td>
      <td>40.831575</td>
      <td>-73.927705</td>
      <td>25-44</td>
      <td>BLACK</td>
      <td>F</td>
    </tr>
    <tr>
      <th>8914831</th>
      <td>05/25/2008</td>
      <td>MISDEMEANOR</td>
      <td>&lt;18</td>
      <td>BLACK</td>
      <td>M</td>
      <td>40.841935</td>
      <td>-73.914246</td>
      <td>45-64</td>
      <td>BLACK</td>
      <td>M</td>
    </tr>
    <tr>
      <th>8914834</th>
      <td>04/16/2008</td>
      <td>FELONY</td>
      <td>25-44</td>
      <td>BLACK</td>
      <td>M</td>
      <td>40.826765</td>
      <td>-73.950203</td>
      <td>45-64</td>
      <td>WHITE</td>
      <td>F</td>
    </tr>
    <tr>
      <th>8914836</th>
      <td>06/25/2008</td>
      <td>MISDEMEANOR</td>
      <td>25-44</td>
      <td>BLACK</td>
      <td>M</td>
      <td>40.578255</td>
      <td>-73.972324</td>
      <td>25-44</td>
      <td>BLACK</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
<p>2071516 rows × 10 columns</p>
</div>

### Idle hands are the Devil's workshop


Generally crime increases during the summer months; however, the reverse is true for minors age < 18. 

During the summer, adults are engaging in less structured social activities - vacations, parties, nightlife, and other outdoof activities. This creates more opportunitiy for crime, whereas work provides a structured activity that decreases crime. 

For minors, however, we see that the structure and supervision they receive at home (or at a summer job) is much better than what they receive at schools. 

<p>
  <img src="notebook/output_18_0.png" width="500"/>
  <img src="notebook/output_21_0.png" width="500"/>
    
</p>

<p>

<img src="notebook/output_21_2.png" width="500"/>
<img src="notebook/output_21_1.png" width="500"/>
<img src="notebook/output_21_3.png" width="500"/>
<img src="notebook/output_21_4.png" width="500"/>

</p>

This is further confirmed if we examine the data by day of the week. For adults, we see higher Felonies and Misdemeanors on the weekends, whether it's summer or not. 

<p>
  <img src="notebook/output_13_0.png" width="500"/>
  <img src="notebook/output_13_1.png" width="500"/>
</p>

But we see the opposite trend for minors. 

<p>
  <img src="notebook/output_14_3.png" width="500"/>
  <img src="notebook/output_14_5.png" width="500"/>
</p>

_Insight_: Structured and supervised environments generally lead to less crime. Families tend to do a better job of creating these environments for kids than schools do. This may be particularly relevant for someone considering the benefits of homeschooling their kids, or weighing the costs and benefits of community activities.

### Profile of a criminal

Criminals tend to attack victims of the same race and the same age group. Also, Felonies and Misdemeanors are comitted by a younger group that Violations. Finally we see a strong negative correlation (corr = -0.92. p = 0.027) between felony and violation percentages among the racial groups. 

  <img src="notebook/output_9_0.png" width="700"/>
  <span> <img src="notebook/output_8_0.png" width="600"/> </span>
<p>
  
<img src="notebook/output_24_0.png" width="500"/>
  
</p>
<img src="notebook/download.png"/>

## Predicting suspect profile based on victim profile

We can leverage these insights to build a classification model using ```RandomForestClassifier``` to predict the profile of the offender based on the victim's age, race, sex, and the type of crime that was committed. This can be potentially beneficial to law enforcement, however we recognize the potential for bias in the data and the risk that this leads to systemic bias in policing. 


```
# Select features and target variables
features = ['VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX', 'LAW_CAT_CD']
X = data_copy[features]
y = data_copy[['SUSP_RACE', 'SUSP_AGE_GROUP', 'SUSP_SEX']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the base model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize the multi-output model
multi_output_model = MultiOutputClassifier(base_model, n_jobs=1)

# Train the model
multi_output_model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = multi_output_model.predict(X_test)

# Evaluate the model for each target variable
accuracy_race = accuracy_score(y_test['SUSP_RACE'], y_pred[:, 0])
accuracy_age_group = accuracy_score(y_test['SUSP_AGE_GROUP'], y_pred[:, 1])
accuracy_sex = accuracy_score(y_test['SUSP_SEX'], y_pred[:, 2])
```
```
Accuracy for SUSP_RACE: 0.72
Accuracy for SUSP_AGE_GROUP: 0.55
Accuracy for SUSP_SEX: 0.74
```



# Analysis: the relationship between crime rates and housing prices

First we use ```Geopandas``` API to attach a zip code to our crime data. Our zillow housing price data and our census data is divided by zip code, so this is essential for connecting our datasets. 

```
def get_zip_code(longitude, latitude, zip_codes):
    
    try: 
        # Create a GeoDataFrame for the input point
        point = gpd.GeoDataFrame(geometry=[Point(longitude, latitude)], crs='EPSG:4326')

        # Perform spatial join to find the corresponding ZIP Code
        joined = gpd.sjoin(point, zip_codes, how='left', predicate='within')

        # Extract the ZIP Code from the resulting GeoDataFrame
        if not joined.empty:
            return joined.iloc[0]['modzcta']  # Use 'modzcta' as the ZIP Code column
        else:
            return None
    except:
        return None

crime_data['zip_code'] = crime_data.apply(lambda row: get_zip_code(row['Longitude'], row['Latitude'], zip_codes), axis=1)
crime_data.to_csv('big_data_w_zipcodes.csv')
```














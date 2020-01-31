# heart_activity_analysis

## Data source
The data used in this project come from https://archive.ics.uci.edu/ml/datasets/Incident+management+process+enriched+event+log.

## Preprocessing
"PPG_preprocessing.py" create the dataset "main_df.csv". "main_df.csv" is already generated in the git repository, so you can pass this step. If you want to execute this code, you must specify the location of the PPG data folder.

## Models creation
"PPG_analysis.py" create the models "Logistic regression", "Random Forest" and "XGBoost" and save it in "models.p". As "main_df.csv", it is already generated.

## Models testing
"flask-api.py" is the api where you can test the models :
  - /logistic_regression 
  - /random_forest
  - /xgboost

You can either call this urls without parameter 
  - it will predict one activity take in the dataset randomly
  
Or you can pass an X and an y (optional) in the request body
  - it will predict the activity and calculate the accuracy if y is specified
  - example: 
### request url 
  http://localhost:5000/random_forest
### request body
{ "X": [ [ -0.9276977768340505, -0.3254098479717742, -0.5122944534774422, 0.7651034025546598, 0.16251075566692375, 0.038958895543037136, -0.13910213016630657, -0.9591819956698056, 0.29796900415615435, -0.10716233976213359, 0.621177125751389, -0.2611527580872457, 0.39406581274431873 ],
 [ -1.1587834604428509, -0.04007687074749291, -1.0910039074752886, -0.7131610905396306, -0.416052156423891, -2.011607644653215, -0.7070919263104573, 1.4022175894884228, 0.17229539267927327, -0.5411065449418008, 0.9477654262801848, 0.9668586236678186, 0.20977207765542003 ],
  [ 0.9209876920363529, -0.4264292361182005, -0.7793911245533713, -0.25271805170698275, 0.008227312442706473, 0.05243064083618694, -0.6356903022742516, 0.7511775169447994, 0.2865442541995149, -0.3614264159236498, -0.39825486799370285, 0.20112471931274933, -0.5519063368412083 ] 
 ], "y": [ 3.0,  6.0,  6.0  ] }

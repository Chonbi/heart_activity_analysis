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
{"X": [[1.0, 49.611369076105795, -0.765625, -0.078125, 0.671875, 34.34, 4.713469, 32.16, 0.8509999513626099, -0.06739997863769531, -0.3694000244140625, -0.08990478515625, -1.47552490234375], [1.0, 72.7882245752842, -0.3125, -0.015625, 1.0, -112.03, 3.051717, 34.16, 0.7574000358581543, -0.11260002851486205, -0.5673999786376953, -0.0998382568359375, -1.38397216796875], [15.0, 62.595231096108144, -0.796875, -0.46875, 0.515625, 53.13, 6.186301, 32.63, 0.7452000379562378, 0.2109999656677246, -0.5074000358581543, 0.6495208740234375, -0.665283203125], [12.0, 75.01180669672969, -0.0625, -0.3125, 0.96875, 112.68, 1.7582099999999998, 34.5, 0.8600000143051147, 0.042799949645996094, -0.3294000029563904, -0.0812530517578125, 0.445556640625], [3.0, 78.96160444397691, -0.46875, -0.078125, 0.875, 37.36, 0.962558, 35.83, 0.8301999568939209, -0.06379997730255127, 0.3997999429702759, -0.2611083984375, -2.545166015625], [7.0, 51.565846519955045, -0.984375, -0.03125, -0.140625, 31.44, 0.147326, 30.77, 0.8711999654769897, -0.1777999997138977, -0.305400013923645, -0.050445556640625, -3.87420654296875]], "y": [0.0, 8.0, 5.0, 0.0, 6.0, 1.0]}
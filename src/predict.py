import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
# import mysql.connector

# Load the trained model
model_filename = "car_prediction_model.pkl"
model = joblib.load(model_filename)

file_path = "src/dataset/CarPrice_Assignment.csv"
dataset = pd.read_csv(file_path)
features = ["CarName", "fueltype", "aspiration", "doornumber", "drivewheel", "enginelocation", "fuelsystem",
            "cylindernumber", "enginetype"]
target = "price"

# Handling missing values and fitting the LabelEncoders for features
dataset[features] = dataset[features].fillna("Unknown")
feature_encoders = {}
for feature in features:
    le = LabelEncoder()
    dataset[feature] = le.fit_transform(dataset[feature])
    feature_encoders[feature] = le


# # Function to encode new data using LabelEncoder for features
# def encode_data(data, feature_encoders):
#     encoded_data = data.copy()
#     for column in data.columns:
#         known_labels = set(feature_encoders[column].classes_)
#         encoded_data[column] = data[column].apply(
#             lambda x: x if x in known_labels else "Unknown"
#         )
#         encoded_data[column] = feature_encoders[column].transform(encoded_data[column])
#     return encoded_data
# Function to encode new data using LabelEncoder for features
def encode_data(data, feature_encoders):
    encoded_data = data.copy()
    for column in data.columns:
        known_labels = set(feature_encoders[column].classes_)
        encoded_data[column] = data[column].apply(
            lambda x: x if x in known_labels else "Unknown"
        )

        # If there are unknown labels, add them to the encoder classes
        unknown_labels = set(encoded_data[column]) - known_labels
        if unknown_labels:
            feature_encoders[column].classes_ = np.append(feature_encoders[column].classes_, list(unknown_labels))

        encoded_data[column] = feature_encoders[column].transform(encoded_data[column])
    return encoded_data



# Predict function
def predict_price(model, data, feature_encoders, team1, team2):
    encoded_data = encode_data(data, feature_encoders)
    prediction = model.predict(encoded_data)
    return team1 if prediction[0] == 1 else team2


# Sample new data for prediction
# new_data = pd.DataFrame(
#     {
#         "city": ["Rajkot"],
#         "team1": ["Gujarat Lions"],
#         "team2": ["Kolkata Knight Riders"],
#         "toss_winner": ["Gujarat Lions"],
#         "toss_decision": ["bat"],
#         "venue": ["Saurashtra Cricket Association Stadium"],
#     }
# )

# team1 = "Gujarat Lions"
# team2 = "Kolkata Knight Riders"

# # Predicting the winner
# predicted_winner = predict_winner(model, new_data, feature_encoders, team1, team2)
# print("Predicted Winner:", predicted_winner)

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) to allow communication with your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

features = ["CarName", "fueltype", "aspiration", "doornumber", "drivewheel", "enginelocation", "fuelsystem",
            "cylindernumber", "enginetype"]

class FormData(BaseModel):
    selectedCarName: str
    selectedFueltype: str
    selectedAspiration: str
    selectedDoornumber: str
    selectedDrivewheel: str
    selectedEnginelocation: str
    selectedFuelsystem: str
    selectedCylinderNumber: str
    selectedEngineType: str

# @app.post("/submit-form")
# async def submit_form(data: FormData):
#     # You can access the form data here and perform further actions (e.g., save to a database)
#     print("Received form data:", data.dict())
    
#     # Perform backend processing here (e.g., save to a database, send to another API, etc.)
    
#     return {"message": "Form submitted successfully"}

# Assuming `predict_winner` is a function that takes your model, data, and other parameters to predict the winner

# @app.post("/submit-form")
# async def submit_form(data: FormData):
#     # Convert FormData instance to dictionary
#     form_data_dict = data.dict()

#     #Rename the keys to match the column names for prediction
#     form_data_dict["city"] = form_data_dict.pop("selectedUmpireA")
#     form_data_dict["team1"] = form_data_dict.pop("selectedTeamA")
#     form_data_dict["team2"] = form_data_dict.pop("selectedTeamB")
#     form_data_dict["toss_winner"] = form_data_dict.pop("selectedTosWinner")
#     form_data_dict["toss_decision"] = form_data_dict.pop("selectedDecicion")
#     form_data_dict["venue"] = form_data_dict.pop("selectedGround")  # Rename selectedUmpireA to venue

#     new_data = pd.DataFrame(
#     {
#         "city": form_data_dict["city"] ,
#         "team1": form_data_dict["team1"],
#         "team2": form_data_dict["team2"] ,
#         "toss_winner":  form_data_dict["toss_winner"],
#         "toss_decision":  form_data_dict["toss_decision"],
#         "venue": form_data_dict["venue"],
#     }
# )

#     # Create a DataFrame from the dictionary
#     # new_data = pd.DataFrame([form_data_dict])
#     print(new_data)

#     # Use the form data for prediction
#     predicted_winner = predict_winner(model, new_data, feature_encoders, form_data_dict["team1"], form_data_dict["team2"])
#     print("Predicted Winner 11:", predicted_winner)

#     # Perform backend processing here (e.g., save to a database, send to another API, etc.)
    
#     return {"message": "Form submitted successfully"}

@app.post("/submit-form")
async def submit_form(data: FormData):

    # Convert FormData instance to dictionary
    form_data_dict = data.dict()

    # Rename the keys to match the column names for prediction
    form_data_dict["CarName"] = form_data_dict.pop("selectedUmpireA")
    form_data_dict["fueltype"] = form_data_dict.pop("selectedTeamA")
    form_data_dict["aspiration"] = form_data_dict.pop("selectedTeamB")
    form_data_dict["doornumber"] = form_data_dict.pop("selectedTosWinner")
    form_data_dict["drivewheel"] = form_data_dict.pop("selectedDecicion")
    form_data_dict["enginelocation"] = form_data_dict.pop("selectedGround")  # Rename selectedUmpireA to venue


    new_data = pd.DataFrame([form_data_dict])
    print(new_data)

    # Use the form data for prediction
    predicted_winner = predict_price(model, new_data, feature_encoders, form_data_dict["CarName"])
    print("Predicted Winner 111:", predicted_winner)

    # Perform backend processing here (e.g., save to a database, send to another API, etc.)
    # Add your logic here

    return {"predicted_winner": predicted_winner}

# @app.get("/get-all-teams")
# async def get_all_teams():
#     connection = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="root1234",
#     database="ipl_cricket"
# )
    cursor = connection.cursor()

    # Example: Execute a query
    cursor.execute("SELECT * FROM teams")

    # Fetch the results
    results = cursor.fetchall()
    return results





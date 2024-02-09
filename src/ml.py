import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings("ignore")


class InputData(BaseModel):
    fuelType: str
    sex: int
    cp: int


# def train_model(inputDate: InputData):
def train_model():


    df = pd.read_csv("src/dataset/CarPrice_Assignment.csv")
    df.isnull().sum()
    df.duplicated().sum()
    df.dtypes
    df.nunique()
    df.describe()
    categorical_columns = [
        "fueltype",
        "aspiration",
        "doornumber",
        "carbody",
        "drivewheel",
        "enginelocation",
        "enginetype",
        "cylindernumber",
        "fuelsystem",
    ]

    for col in categorical_columns:
        print(f"Category in {col} is : {df[col].unique()}")

    numerical_features = [
        "wheelbase",
        "carlength",
        "carwidth",
        "carheight",
        "curbweight",
        "enginesize",
        "boreratio",
        "stroke",
        "compressionratio",
        "horsepower",
        "peakrpm",
        "citympg",
        "highwaympg",
        "price",
    ]

    plt.figure(figsize=(12, 8))
    for feature in numerical_features:
        plt.subplot(3, 5, numerical_features.index(feature) + 1)
        sns.histplot(data=df[feature], bins=20, kde=True)
        plt.title(feature)
    plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(data=df["price"], bins=20, kde=True)
    plt.title("Distribution of Price")
    # plt.show()

    categorical_columns = [
        "fueltype",
        "aspiration",
        "doornumber",
        "carbody",
        "drivewheel",
        "enginelocation",
        "enginetype",
        "cylindernumber",
        "fuelsystem",
    ]

    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
    axes = axes.ravel()  # Flatten the 2D array of axes

    # Loop through each categorical column
    for i, column in enumerate(categorical_columns):
        sns.countplot(
            x=df[column], data=df, palette="bright", ax=axes[i], saturation=0.95
        )
        for container in axes[i].containers:
            axes[i].bar_label(container, color="black", size=10)
        axes[i].set_title(f"Count Plot of {column.capitalize()}")
        axes[i].set_xlabel(column.capitalize())
        axes[i].set_ylabel("Count")

    # Adjust layout and show plots
    plt.tight_layout()
    # plt.show()

    n = 20  # Number of top car models to plot
    top_car_models = df["CarName"].value_counts().head(n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_car_models.values, y=top_car_models.index)
    plt.title(f"Top {n} Car Models by Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Car Model")
    plt.tight_layout()
    # plt.show()

    avg_prices_by_car = (
        df.groupby("CarName")["price"].mean().sort_values(ascending=False)
    )

    # Plot top N car models by average price
    n = 20  # Number of top car models to plot
    top_car_models = avg_prices_by_car.head(n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_car_models.values, y=top_car_models.index)
    plt.title(f"Top {n} Car Models by Average Price")
    plt.xlabel("Average Price")
    plt.ylabel("Car Model")
    plt.tight_layout()
    plt.show()

    # Categorical Feature vs. Price
    plt.figure(figsize=(12, 8))
    for feature in categorical_columns:
        plt.subplot(3, 3, categorical_columns.index(feature) + 1)
        sns.boxplot(data=df, x=feature, y="price")
        plt.title(f"{feature} vs. Price")
    plt.tight_layout()
    plt.show()

    # Correlation Analysis
    correlation_matrix = df[numerical_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Extract brand and model from CarName
    df["brand"] = df["CarName"].apply(lambda x: x.split(" ")[0])
    df["model"] = df["CarName"].apply(lambda x: " ".join(x.split(" ")[1:]))

    # Define categorical and numerical columns
    categorical_columns = [
        "fueltype",
        "aspiration",
        "doornumber",
        "carbody",
        "drivewheel",
        "enginelocation",
        "enginetype",
        "cylindernumber",
        "fuelsystem",
        "brand",
        "model",
    ]
    numerical_columns = [
        "wheelbase",
        "carlength",
        "carwidth",
        "carheight",
        "curbweight",
        "enginesize",
        "boreratio",
        "stroke",
        "compressionratio",
        "horsepower",
        "peakrpm",
        "citympg",
        "highwaympg",
    ]

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Feature engineering
    df["power_to_weight_ratio"] = df["horsepower"] / df["curbweight"]
    for column in numerical_columns:
        df[f"{column}_squared"] = df[column] ** 2
    df["log_enginesize"] = np.log(df["enginesize"] + 1)

    # Feature scaling
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Splitting the dataset
    X = df.drop(
        ["price", "CarName"], axis=1
    )  # Include the engineered features and CarName
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2_square = r2_score(y_test, y_pred)
    print(f" R-squared: {r2_square}")
    print(f"Mean Squared Error: {mse}")

    # Saving the trained model to a file
    model_filename = "car_prediction_model.pkl"
    joblib.dump(model, model_filename)

    # pred_df = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred, 'Difference': y_test - y_pred})
    # pred_df
    # print(pred_df)
    return "Done"

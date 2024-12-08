from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os
import pandas as pd


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def clean_data(X, y):
    data = X.copy()
    data['selling_price'] = y
    for col in data.columns:
        try:
            median_value = data[col].median()
        except:
            median_value = data.mode()[col][0]
        data[col].fillna(median_value, inplace=True)
        data[col] = data[col].replace(np.nan, median_value)
        col_duplicates = data.columns.difference(['selling_price'])
        data = data.drop_duplicates(subset=col_duplicates, keep='first')
    def get_float(x):
        try:
            return float(x.split(" ")[0])
        except:
            return 0
    data['mileage'] = data['mileage'].apply(lambda a: get_float(a))
    data['engine'] = data['engine'].apply(lambda a: get_float(a))
    data['max_power'] = data['max_power'].apply(lambda a: get_float(a))
    data = data.drop(['torque'], axis = 1)
    data['engine'] = data['engine'].apply(lambda a: int(a))
    data['seats'] = data['seats'].apply(lambda a: int(a))

    data = data.drop(['name', 'fuel', 'seller_type', 'transmission', 'owner'], axis = 1)
    return [data.drop('selling_price', axis = 1), data['selling_price']]


# df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
# df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    if not os.path.exists('car_price_model.pkl'):
        return 0
    with open('car_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    df = pd.DataFrame([item.dict()])
    X, y = clean_data(df.drop('selling_price', axis = 1), df['selling_price'])
    return model.predict(X)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    if not os.path.exists('car_price_model.pkl'):
        return 0
    result = []
    for item in items:
        pred = predict_item(item)
        result.append(pred)
    return result

@app.post("/fit_model")
def fit_model(csv_file_path: str):
    if not os.path.exists(csv_file_path):
        return {"error": "File not found"}
    df = pd.read_csv(csv_file_path)
    X = df.drop('selling_price',axis = 1)
    y = df['selling_price']
    X, y = clean_data(X, y)
    model = LinearRegression()
    model.fit(X, y)
    with open('car_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return {"message": "Модель прошла тренировку"}
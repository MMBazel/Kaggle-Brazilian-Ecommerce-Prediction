# TODO: Add docstrings
# TODO: Make sure to import the other modules
# TODO: Check for any missing docstrings


import pandas as pd
import polars as pl

# from datetime import datetime, timedelta
# import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import os

# from time import time
from time import perf_counter


def read_files(full_path, dataset_dict):
    # given the dataset_dict, get the value correspondign to a key, apply the unzip_files function to it
    target_directory = f"{full_path}/data/raw"

    customers_df = unzip_files(dataset_dict.get("customers_dataset"), target_directory)

    customers_df = pl.scan_csv(
        f"{full_path}/data/raw/olist_customers_dataset.csv"
    ).collect()

    sellers_df = pl.scan_csv(
        f"{full_path}/data/raw/olist_sellers_dataset.csv"
    ).collect()

    ord_items_df = pl.scan_csv(
        f"{full_path}/data/raw/olist_order_items_dataset.csv"
    ).collect()

    products_df = pl.scan_csv(
        f"{full_path}/data/raw/olist_products_dataset.csv"
    ).collect()

    orders_df = pl.scan_csv(f"{full_path}/data/raw/olist_orders_dataset.csv").collect()

    payments_df = pl.scan_csv(
        f"{full_path}/data/raw/olist_order_payments_dataset.csv"
    ).collect()

    brazilian_holidays_df = pl.scan_csv(
        f"{full_path}/data/raw/Brazilian_Holidays.csv"
    ).collect()

    brazilian_cities_population_df = pl.scan_csv(
        f"{full_path}/data/raw/Population_Brazilian_Cities_V2.csv"
    ).collect()

    return (
        customers_df,
        sellers_df,
        ord_items_df,
        products_df,
        orders_df,
        payments_df,
        brazilian_holidays_df,
        brazilian_cities_population_df,
    )


def model_train(final_dataset_pd):
    X = final_dataset_pd.drop(
        [
            "label_estimated_delivery_duration",
            "label_actual_delivery_duration",
            "order_purchase_timestamp",
            "feat_customer_state",
            "feat_customer_city",
        ],
        axis=1,
    )
    y = final_dataset_pd[["label_actual_delivery_duration"]]

    print("=====> =====> Splitting train_test")

    rf = RandomForestRegressor()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    print("=====> =====> Fitting Random Forest Regressions Model")

    rf.fit(X_train, y_train)

    print("=====> =====> Scoring Random Forest Regression Model")

    score = rf.score(X_val, y_val)

    print("=====> =====> Random Forest score:", score)

    return rf


def close_out_pipeline(rf, final_dataset_pd):

    final_dataset_pd["predictions"] = rf.predict(
        final_dataset_pd[
            [
                "feat_total_payment_value",
                "feat_num_items_per_order",
                "feat_num_cat_per_order",
                "feat_num_sellers_per_order",
                "feat_num_seller_cities_per_order",
                "feat_customer_holiday_impact",
                "feat_seller_holiday_impact",
            ]
        ]
    )

    full_path = os.getcwd()
    final_dataset_pd.to_csv(f"{full_path}/data/backup/predictions.csv")

    pickle.dump(rf, open(f"{full_path}/model/model.pkl", "wb"))

import pandas as pd
import polars as pl
from datetime import datetime, timedelta
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import os
from time import time
from time import perf_counter

import shutil

def unzip_files(full_path):
    filenames = [
        "/data/raw/olist_customers_dataset.csv.zip",
        "/data/raw/olist_sellers_dataset.csv.zip",
        "/data/raw/olist_order_items_dataset.csv.zip",
        "/data/raw/olist_products_dataset.csv.zip",
        "/data/raw/olist_orders_dataset.csv.zip",
        "/data/raw/olist_order_payments_dataset.csv.zip",
    ]
    for file in filenames:
        shutil.unpack_archive(f"{full_path}{file}", f"{full_path}/data/raw/")

def read_files(full_path):
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

    orders_df = pl.scan_csv(
        f"{full_path}/data/raw/olist_orders_dataset.csv"
    ).collect()

    payments_df = pl.scan_csv(
        f"{full_path}/data/raw/olist_order_payments_dataset.csv"
    ).collect()

    brazilian_holidays_df = pl.scan_csv(
        f"{full_path}/data/raw/Brazilian_Holidays.csv"
    ).collect()

    brazilian_cities_population_df = pl.scan_csv(
    f"{full_path}/data/raw/Population_Brazilian_Cities_V2.csv"
    ).collect()

    return customers_df, sellers_df, ord_items_df, products_df, orders_df, payments_df, brazilian_holidays_df, brazilian_cities_population_df

def get_feature_total_payment_value(payments_df):
    return payments_df.lazy().select(pl.col(["order_id", "payment_value"])).rename({"payment_value": "feat_total_payment_value"}).collect()

def get_feature_num_items_per_order(ord_items_df):
    return (
        ord_items_df.lazy()
        .select(
            [
                pl.col("order_id"),
                pl.col("order_item_id")
                .unique()
                .count()
                .over("order_id")
                .alias("feat_num_items_per_order"),
            ]
        )
        .collect()
)

def get_feature_num_cat_per_order(ord_items_df, products_df): 
    return ord_items_df.lazy().join(products_df.lazy(), how="left", on="product_id").select(pl.col(["order_id", "order_item_id", "product_category_name"])).select(
        [
            pl.col("order_id"),
            pl.col("product_category_name")
            .unique()
            .count()
            .over("order_id")
            .alias("feat_num_cat_per_order"),
        ]
    ).collect()

def get_feature_customer_locat(orders_df, customers_df):
    return orders_df.lazy().select(pl.col(["order_id", "customer_id"])).join(
            customers_df.lazy().select(
                pl.col(["customer_id", "customer_city", "customer_state"])
            ),
            how="left",
            on="customer_id",
        ).rename(
            {"customer_city": "feat_customer_city", "customer_state": "feat_customer_state"}
        ).collect()

def get_feature_num_sellers(ord_items_df, sellers_df):
    return ord_items_df.lazy().join(sellers_df.lazy(), how="left", on="seller_id").select(
        [
            pl.col("order_id"),
            pl.col("seller_id")
            .unique()
            .count()
            .over("order_id")
            .alias("feat_num_sellers_per_order"),
        ]
    ).collect()

def get_feature_num_cities(ord_items_df, sellers_df):
    return ord_items_df.lazy().join(sellers_df.lazy(), how="left", on="seller_id").select(
        [
            pl.col("order_id"),
            pl.col("seller_city")
            .unique()
            .count()
            .over("order_id")
            .alias("feat_num_seller_cities_per_order"),
        ]
    ).collect()

def prepare_holiday_data(brazilian_holidays_df):
    brazilian_holidays_transformed_df = brazilian_holidays_df.with_columns(
        [pl.col("Holiday_Date").str.strptime(pl.Date, fmt="%m/%d/%Y").cast(pl.Datetime)]
    ).with_columns(
        [
            (pl.col("Holiday_Date") - pl.duration(days=3)).alias("Holiday_Week_Start"),
            (pl.col("Holiday_Date") + pl.duration(days=3)).alias("Holiday_Week_End"),
        ]
    )

    national_holidays_df = brazilian_holidays_transformed_df.filter(
        pl.col("Type") == "National"
    ).select(
        [
            pl.col("Type"),
            pl.col("State"),
            pl.col("City_Lower"),
            pl.col("Holiday_Week_Start"),
            pl.col("Holiday_Week_End"),
        ]
    )

    state_holidays_df = brazilian_holidays_transformed_df.filter(
        pl.col("Type") == "State"
    ).select(
        [
            pl.col("Type"),
            pl.col("State"),
            pl.col("City_Lower"),
            pl.col("Holiday_Week_Start"),
            pl.col("Holiday_Week_End"),
        ]
    )

    city_holidays_df = brazilian_holidays_transformed_df.filter(
        pl.col("Type") == "City"
    ).select(
        [
            pl.col("Type"),
            pl.col("State"),
            pl.col("City_Lower"),
            pl.col("Holiday_Week_Start"),
            pl.col("Holiday_Week_End"),
        ]
    )

    return national_holidays_df, state_holidays_df, city_holidays_df

def register_initial_SQL_Context(national_holidays_df, state_holidays_df, city_holidays_df):
    sql = pl.SQLContext()
    sql.register("national_holidays_df", national_holidays_df.lazy())
    sql.register("state_holidays_df", state_holidays_df.lazy())
    sql.register("city_holidays_df", city_holidays_df.lazy())

    return sql

def get_customer_regions_by_order(orders_df, customers_df):
    return orders_df.lazy().select(pl.col(["order_id", "customer_id", "order_purchase_timestamp"])).join(
            customers_df.lazy().select(
                pl.col(["customer_id", "customer_city", "customer_state"])
            ),
            how="left",
            on="customer_id",
        ).with_columns(
            pl.col("order_purchase_timestamp")
            .str.strptime(pl.Date, fmt="%Y-%m-%d %H:%M:%S")
            .cast(pl.Datetime)
        ).collect()

def query_create_customer_holiday(sql, get_customer_regions_by_order_df):

    sql.register("get_customer_regions_by_order", get_customer_regions_by_order_df.lazy())

    isNationalHolidayCustomerOrder = sql.query(
        """
        Select 
            order_id,
            order_purchase_timestamp,
            customer_id as user_id,
            customer_city as city,
            customer_state as state,
            Holiday_Week_Start,
            Holiday_Week_End,
            1 as isHoliday
        From get_customer_regions_by_order
        cross join national_holidays_df
        where order_purchase_timestamp >= Holiday_Week_Start and Holiday_Week_End >= order_purchase_timestamp
        """
    )

    isStateHolidayCustomerOrder = sql.query(
        """
        Select 
            order_id,
            order_purchase_timestamp,
            customer_id as user_id,
            customer_city as city,
            customer_state as state,
            Holiday_Week_Start,
            Holiday_Week_End,
            1 as isHoliday
        From get_customer_regions_by_order
        join state_holidays_df
        on state_holidays_df.State = get_customer_regions_by_order.customer_state
        where order_purchase_timestamp >= Holiday_Week_Start and Holiday_Week_End >= order_purchase_timestamp
        """
    )

    isCityHolidayCustomerOrder = sql.query(
        """
        Select 
            order_id,
            order_purchase_timestamp,
            customer_id as user_id,
            customer_city as city,
            customer_state as state,
            Holiday_Week_Start,
            Holiday_Week_End,
            1 as isHoliday
        From get_customer_regions_by_order
        join city_holidays_df
        on city_holidays_df.City_Lower = get_customer_regions_by_order.customer_city
        where order_purchase_timestamp >= Holiday_Week_Start and Holiday_Week_End >= order_purchase_timestamp
        """
    )

    return pl.concat(
        [
            isNationalHolidayCustomerOrder,
            isStateHolidayCustomerOrder,
            isCityHolidayCustomerOrder,
        ]
    )

def get_feature_customer_order_holidays(orders_df, allHolidaysByOrderCustomer):
    return (
        orders_df.select(pl.col("order_id"))
        .join(
            allHolidaysByOrderCustomer.select(pl.col(["order_id", "isHoliday"])),
            how="left",
            on="order_id",
        )
        .groupby(by="order_id")
        .agg(
            [
                pl.col("isHoliday").sum().alias("feat_customer_holiday_impact"),
            ]
        )
        .with_columns(pl.col("feat_customer_holiday_impact").fill_null(0))
        .sort("feat_customer_holiday_impact", descending=True)
    )

def get_seller_regions_by_order(orders_df, ord_items_df, sellers_df):
    return orders_df.lazy().select(pl.col(["order_id", "order_purchase_timestamp"])).join(
            ord_items_df.lazy().select(pl.col(["order_id", "order_item_id", "seller_id"])),
            how="left",
            on="order_id",
        ).join(
            sellers_df.lazy().select(pl.col(["seller_id", "seller_city", "seller_state"])),
            how="left",
            on="seller_id",
        ).with_columns(
            pl.col("order_purchase_timestamp")
            .str.strptime(pl.Date, fmt="%Y-%m-%d %H:%M:%S")
            .cast(pl.Datetime)
        ).collect()

def query_create_seller_holiday(sql, get_sellers_regions_by_order_df):
    sql.register("get_sellers_regions_by_order", get_sellers_regions_by_order_df.lazy())

    isNationalHolidaySellerOrder = sql.query(
        """
        Select 
            order_id,
            order_purchase_timestamp,
            seller_id as user_id,
            seller_city as city,
            seller_state as state,
            Holiday_Week_Start,
            Holiday_Week_End,
            1 as isHoliday
        From get_sellers_regions_by_order
        cross join national_holidays_df
        where order_purchase_timestamp >= Holiday_Week_Start and Holiday_Week_End >= order_purchase_timestamp
        """
    )

    isStateHolidaySellerOrder = sql.query(
        """
        Select 
            order_id,
            order_purchase_timestamp,
            seller_id as user_id,
            seller_city as city,
            seller_state as state,
            Holiday_Week_Start,
            Holiday_Week_End,
            1 as isHoliday
        From get_sellers_regions_by_order
        join state_holidays_df
        on state_holidays_df.State = get_sellers_regions_by_order.seller_state
        where order_purchase_timestamp >= Holiday_Week_Start and Holiday_Week_End >= order_purchase_timestamp
        """
    )

    isCityHolidaySellerOrder = sql.query(
        """
        Select 
            order_id,
            order_purchase_timestamp,
            seller_id as user_id,
            seller_city as city,
            seller_state as state,
            Holiday_Week_Start,
            Holiday_Week_End,
            1 as isHoliday
        From get_sellers_regions_by_order
        join city_holidays_df
        on city_holidays_df.City_Lower = get_sellers_regions_by_order.seller_city
        where order_purchase_timestamp >= Holiday_Week_Start and Holiday_Week_End >= order_purchase_timestamp
        """
    )

    return pl.concat([isNationalHolidaySellerOrder, isStateHolidaySellerOrder, isCityHolidaySellerOrder])

def get_feature_seller_order_holidays(orders_df, allHolidaysByOrderSeller):
    return orders_df.select(pl.col("order_id")).join(
            allHolidaysByOrderSeller.select(pl.col(["order_id", "isHoliday"])),
            how="left",
            on="order_id",
        ).groupby(by="order_id").agg(
            [
                pl.col("isHoliday").sum().alias("feat_seller_holiday_impact"),
            ]
        ).with_columns(pl.col("feat_seller_holiday_impact").fill_null(0)).sort("feat_seller_holiday_impact", descending=True)

def create_labels(orders_df):
    orders_pd = orders_df.select(
        pl.col(
            [
                "order_id",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ]
        )
    ).to_pandas()

    orders_pd[["order_delivered_customer_date", "order_purchase_timestamp"]] = orders_pd[["order_delivered_customer_date", "order_purchase_timestamp"]].apply(pd.to_datetime)
    
    orders_pd["order_estimated_delivery_date"] = pd.to_datetime(orders_pd["order_estimated_delivery_date"], format="%Y-%m-%d %H:%M:%S")

    orders_pd["label_actual_delivery_duration"] = (orders_pd["order_delivered_customer_date"] - orders_pd["order_purchase_timestamp"]).apply(lambda x: x / pd.Timedelta(1, "h"))
    
    orders_pd["label_estimated_delivery_duration"] = (orders_pd["order_estimated_delivery_date"] - orders_pd["order_purchase_timestamp"]).apply(lambda x: x / pd.Timedelta(1, "h"))

    return pl.from_pandas(orders_pd)

def create_final_dataset(pay_df, num_items_per_order_df, num_cat_per_order_df, customer_locat_df, num_sellers_per_order_df, num_cities_per_order_df, orderHolidayPurchaseCustomer, orderHolidayPurchaseSeller, delivery_duration_df):
    final_dataset = (pay_df.lazy()).join(num_items_per_order_df.lazy(), how="left", on="order_id").join(num_cat_per_order_df.lazy(), how="left", on="order_id").join(customer_locat_df.lazy(), how="left", on="order_id").join(num_sellers_per_order_df.lazy(), how="left", on="order_id").join(num_cities_per_order_df.lazy(), how="left", on="order_id").join(orderHolidayPurchaseCustomer.lazy(), how="left", on="order_id").join(orderHolidayPurchaseSeller.lazy(), how="left", on="order_id").join(delivery_duration_df.lazy(), how="left", on="order_id")    
    final_dataset = final_dataset.collect()
    final_dataset.write_csv("data/backup/exploratory_data_set.csv")

    return final_dataset.drop(["order_estimated_delivery_date","order_delivered_customer_date","customer_id","order_id",]).drop_nulls().to_pandas()

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


def main():
    full_path = os.getcwd()

    time_dict = {}


    
    print("\n\n----------------------------- 📁 Reading Files From Repository 📁 ----------------------------- \n\n")
    
    start_time = perf_counter()
    unzip_files(full_path)
    stop_time = perf_counter()
    time_dict["Unzipping files"] = [
        "Stage",
        (stop_time - start_time) / 60,
    ]
    
    start_time = perf_counter()
    customers_df, sellers_df, ord_items_df, products_df, orders_df, payments_df, brazilian_holidays_df, brazilian_cities_population_df = read_files(full_path)
    stop_time = perf_counter()
    time_dict["Reading Files From Repository"] = ["Stage",(stop_time-start_time)/60]

    
    print("----------------------------- 🛠️ Creating Features 🧰 ----------------------------- \n\n")
    
    print("Creating Feature 1️⃣: Total Payment Value\n\n")
    step_start_time = perf_counter()
    pay_df = get_feature_total_payment_value(payments_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 1: Total Payment Value"] = ["Step", (step_end_time-step_start_time)/60]

    print("Creating Feature 2️⃣: Num Items Per Order\n\n")
    step_start_time = perf_counter()
    num_items_per_order_df = get_feature_num_items_per_order(ord_items_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 2: Num Items Per Order"] = ["Step",(step_end_time-step_start_time)/60]


    print("Creating Feature 3️⃣: Num Categories Per Order\n\n")
    step_start_time = perf_counter()
    num_cat_per_order_df = get_feature_num_cat_per_order(ord_items_df, products_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 3: Num Categories Per Order"] = ["Step",(step_end_time-step_start_time)/60]


    print("Creating Feature 4️⃣: Customer Location\n\n")
    step_start_time = perf_counter()
    customer_locat_df = get_feature_customer_locat(orders_df, customers_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 4: Customer Location"] = ["Step",(step_end_time-step_start_time)/60]


    print("Creating Feature 5️⃣: Num Sellers Per Order\n\n")
    step_start_time = perf_counter()
    num_sellers_per_order_df = get_feature_num_sellers(ord_items_df, sellers_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 5: Num Sellers Per Order"] = ["Step",(step_end_time-step_start_time)/60]


    print("Creating Feature 6️⃣: Num Seller Cities Per Order\n\n")
    step_start_time = perf_counter()
    num_cities_per_order_df = get_feature_num_cities(ord_items_df, sellers_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 6: Num Seller Cities Per Order"] = ["Step",(step_end_time-step_start_time)/60]


    print("Creating Feature 7️⃣: Num Holidays For Customer + Feature 8️⃣: Num Holidays For Sellers\n\n")

    print("=====> Step 1: Start processing brazilian holidays")
    step_sub_start_time = perf_counter()
    national_holidays_df, state_holidays_df, city_holidays_df = prepare_holiday_data(brazilian_holidays_df)
    step_sub_end_time = perf_counter()
    time_dict["Step 1: Start processing brazilian holidays"] = ["Sub-Step",(step_sub_end_time-step_sub_start_time)/60]

    print("=====> Step 2: Register SQl Context")
    step_sub_start_time = perf_counter()
    sql = register_initial_SQL_Context(national_holidays_df, state_holidays_df, city_holidays_df)
    step_sub_end_time = perf_counter()
    time_dict["Step 2: Register SQl Context"] = ["Sub-Step",(step_sub_end_time-step_sub_start_time)/60]


    print("=====> Step 3: Prepare & Create Feature 7️⃣: Num Holidays For Customer")
    step_sub_start_time = perf_counter()
    get_customer_regions_by_order_df = get_customer_regions_by_order(orders_df, customers_df)
    allHolidaysByOrderCustomer = query_create_customer_holiday(sql, get_customer_regions_by_order_df)
    orderHolidayPurchaseCustomer = get_feature_customer_order_holidays(orders_df, allHolidaysByOrderCustomer)
    step_sub_end_time = perf_counter()
    time_dict["Step 3: Prepare & Create Feature 7: Num Holidays For Customer"] = ["Sub-Step",(step_sub_end_time-step_sub_start_time)/60]


    print("=====> Step 4: Prepare & Create Feature 8️⃣: Num Holidays For Sellers\n\n")
    step_start_time = perf_counter()
    get_sellers_regions_by_order_df = get_seller_regions_by_order(orders_df, ord_items_df, sellers_df)
    allHolidaysByOrderSeller = query_create_seller_holiday(sql, get_sellers_regions_by_order_df)
    orderHolidayPurchaseSeller = get_feature_seller_order_holidays(orders_df, allHolidaysByOrderSeller)
    step_sub_end_time = perf_counter()
    time_dict["Step 4: Prepare & Create Feature 8: Num Holidays For Sellers"] = ["Sub-Step",(step_sub_end_time-step_sub_start_time)/60]


    print("\n----------------------------- 🏷️ Create Label: Actual Delivery Duration 📦----------------------------- \n\n")
    start_time = perf_counter()
    delivery_duration_df = create_labels(orders_df)
    stop_time = perf_counter()
    time_dict["Creating Features"] = ["Stage",(stop_time-start_time)/60]



    print("----------------------------- 🚂 Preparing to Train Random Forest Regression Model 🌴----------------------------- \n\n")

    print("=====> Step 1: Create Final Dataset\n")
    step_start_time = perf_counter()
    final_dataset_pd = create_final_dataset(pay_df, num_items_per_order_df, num_cat_per_order_df, customer_locat_df, num_sellers_per_order_df, num_cities_per_order_df, orderHolidayPurchaseCustomer, orderHolidayPurchaseSeller, delivery_duration_df)
    step_end_time = perf_counter()
    time_dict["Step 1: Create Final Dataset"] = ["Step",(step_end_time-step_start_time)/60]


    print("=====> Step 2: Train Model")
    step_start_time = perf_counter()
    rf = model_train(final_dataset_pd)
    step_end_time = perf_counter()
    time_dict["Step 2: Train Model"] = ["Step",(step_end_time-step_start_time)/60]


    print("\n\n-----------------------------🫖 Close out pipeline 🫖----------------------------- \n\n")
    start_time = perf_counter()

    print("🥒 Pickle model 🫙\n")
    print("✍️ Write predictions 🪄")

    close_out_pipeline(rf, final_dataset_pd)
    stop_time = perf_counter()
    time_dict["Close out pipeline"] = ["Stage",(stop_time-start_time)/60]


    profile_log_df = pd.DataFrame.from_dict(time_dict,orient="index")
    profile_log_df.columns=["Type","Time in Mins"]
    print(profile_log_df)
    profile_log_df.to_csv('profile_report.csv')

    print("\n\n----------------------------- 🏎️ Start Streamlit! 📊----------------------------- \n\n")



if __name__ == "__main__":
    main()

    print("\n\n=============================================================")
    print("📌 At any time you can stop the server with Ctrl+c 📌")
    print("=============================================================\n\n")

    os.system("streamlit run dashboard.py")

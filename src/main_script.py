# TODO: Add docstring
# TODO: Check for any missing import statements
# TODO: Check if there's a way to write decorators to time the different steps
from dataprep.dataloader import read_files
import featurization.create_features as cf
import featurization.holiday_helper as hh
import training.training_prep as tp
import training.model_training as mt

import pandas as pd

import os

# from time import time
from time import perf_counter


def main():
    full_path = os.getcwd()

    time_dict = {}

    print("\n\n 📁 Reading Files From Repository 📁 \n\n")
    start_time = perf_counter()
    (
        customers_df,
        sellers_df,
        ord_items_df,
        products_df,
        orders_df,
        payments_df,
        brazilian_holidays_df,
    ) = read_files(full_path)
    stop_time = perf_counter()
    time_dict["Reading Files From Repository"] = [
        "Stage",
        (stop_time - start_time) / 60,
    ]

    print(
        "----------------------------- 🛠️ Creating Features 🧰 ----------------------------- \n\n"
    )

    print("Creating Feature 1️⃣: Total Payment Value\n\n")
    step_start_time = perf_counter()
    pay_df = cf.get_feature_total_payment_value(payments_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 1: Total Payment Value"] = [
        "Step",
        (step_end_time - step_start_time) / 60,
    ]

    print("Creating Feature 2️⃣: Num Items Per Order\n\n")
    step_start_time = perf_counter()
    num_items_per_order_df = cf.get_feature_num_items_per_order(ord_items_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 2: Num Items Per Order"] = [
        "Step",
        (step_end_time - step_start_time) / 60,
    ]

    print("Creating Feature 3️⃣: Num Categories Per Order\n\n")
    step_start_time = perf_counter()
    num_cat_per_order_df = cf.get_feature_num_cat_per_order(ord_items_df, products_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 3: Num Categories Per Order"] = [
        "Step",
        (step_end_time - step_start_time) / 60,
    ]

    print("Creating Feature 4️⃣: Customer Location\n\n")
    step_start_time = perf_counter()
    customer_locat_df = cf.get_feature_customer_locat(orders_df, customers_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 4: Customer Location"] = [
        "Step",
        (step_end_time - step_start_time) / 60,
    ]

    print("Creating Feature 5️⃣: Num Sellers Per Order\n\n")
    step_start_time = perf_counter()
    num_sellers_per_order_df = cf.get_feature_num_sellers(ord_items_df, sellers_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 5: Num Sellers Per Order"] = [
        "Step",
        (step_end_time - step_start_time) / 60,
    ]

    print("Creating Feature 6️⃣: Num Seller Cities Per Order\n\n")
    step_start_time = perf_counter()
    num_cities_per_order_df = cf.get_feature_num_cities(ord_items_df, sellers_df)
    step_end_time = perf_counter()
    time_dict["Creating Feature 6: Num Seller Cities Per Order"] = [
        "Step",
        (step_end_time - step_start_time) / 60,
    ]

    print(
        "Creating Feature 7️⃣: Num Holidays For Customer + Feature 8️⃣: Num Holidays For Sellers\n\n"
    )

    print("=====> Step 1: Start processing brazilian holidays")
    step_sub_start_time = perf_counter()
    national_holidays_df, state_holidays_df, city_holidays_df = cf.prepare_holiday_data(
        brazilian_holidays_df
    )
    step_sub_end_time = perf_counter()
    time_dict["Step 1: Start processing brazilian holidays"] = [
        "Sub-Step",
        (step_sub_end_time - step_sub_start_time) / 60,
    ]

    print("=====> Step 2: Register SQl Context")
    step_sub_start_time = perf_counter()
    sql = hh.register_initial_SQL_Context(
        national_holidays_df, state_holidays_df, city_holidays_df
    )
    step_sub_end_time = perf_counter()
    time_dict["Step 2: Register SQl Context"] = [
        "Sub-Step",
        (step_sub_end_time - step_sub_start_time) / 60,
    ]

    print("=====> Step 3: Prepare & Create Feature 7️⃣: Num Holidays For Customer")
    step_sub_start_time = perf_counter()
    get_customer_regions_by_order_df = cf.get_customer_regions_by_order(
        orders_df, customers_df
    )
    allHolidaysByOrderCustomer = hh.query_create_customer_holiday(
        sql, get_customer_regions_by_order_df
    )
    orderHolidayPurchaseCustomer = cf.get_feature_customer_order_holidays(
        orders_df, allHolidaysByOrderCustomer
    )
    step_sub_end_time = perf_counter()
    time_dict["Step 3: Prepare & Create Feature 7: Num Holidays For Customer"] = [
        "Sub-Step",
        (step_sub_end_time - step_sub_start_time) / 60,
    ]

    print("=====> Step 4: Prepare & Create Feature 8️⃣: Num Holidays For Sellers\n\n")
    step_start_time = perf_counter()
    get_sellers_regions_by_order_df = cf.get_seller_regions_by_order(
        orders_df, ord_items_df, sellers_df
    )
    allHolidaysByOrderSeller = hh.query_create_seller_holiday(
        sql, get_sellers_regions_by_order_df
    )
    orderHolidayPurchaseSeller = cf.get_feature_seller_order_holidays(
        orders_df, allHolidaysByOrderSeller
    )
    step_sub_end_time = perf_counter()
    time_dict["Step 4: Prepare & Create Feature 8: Num Holidays For Sellers"] = [
        "Sub-Step",
        (step_sub_end_time - step_sub_start_time) / 60,
    ]

    print(
        "\n----------------------------- 🏷️ Create Label: Actual Delivery Duration 📦----------------------------- \n\n"
    )
    start_time = perf_counter()
    delivery_duration_df = create_labels(orders_df)
    stop_time = perf_counter()
    time_dict["Creating Features"] = ["Stage", (stop_time - start_time) / 60]

    print(
        "----------------------------- 🚂 Preparing to Train Random Forest Regression Model 🌴----------------------------- \n\n"
    )

    print("=====> Step 1: Create Final Dataset\n")
    step_start_time = perf_counter()
    final_dataset_pd = tp.create_final_dataset(
        pay_df,
        num_items_per_order_df,
        num_cat_per_order_df,
        customer_locat_df,
        num_sellers_per_order_df,
        num_cities_per_order_df,
        orderHolidayPurchaseCustomer,
        orderHolidayPurchaseSeller,
        delivery_duration_df,
    )
    step_end_time = perf_counter()
    time_dict["Step 1: Create Final Dataset"] = [
        "Step",
        (step_end_time - step_start_time) / 60,
    ]

    print("=====> Step 2: Train Model")
    step_start_time = perf_counter()
    rf = mt.model_train(final_dataset_pd)
    step_end_time = perf_counter()
    time_dict["Step 2: Train Model"] = ["Step", (step_end_time - step_start_time) / 60]

    print(
        "\n\n-----------------------------🫖 Close out pipeline 🫖----------------------------- \n\n"
    )
    start_time = perf_counter()

    print("🥒 Pickle model 🫙\n")
    print("✍️ Write predictions 🪄")

    mt.close_out_pipeline(rf, final_dataset_pd)
    stop_time = perf_counter()
    time_dict["Close out pipeline"] = ["Stage", (stop_time - start_time) / 60]

    profile_log_df = pd.DataFrame.from_dict(time_dict, orient="index")
    profile_log_df.columns = ["Type", "Time in Mins"]
    print(profile_log_df)
    profile_log_df.to_csv("profile_report.csv")

    print("\n\n 🏎️ Start Streamlit! 📊 \n\n")


if __name__ == "__main__":
    main()

    print("\n\n=============================================================")
    print("📌 At any time you can stop the server with Ctrl+c 📌")
    print("=============================================================\n\n")

    os.system("streamlit run dashboard.py")

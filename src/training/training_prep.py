# TODO: Addd missing import statements
# TODO : Add docstrings

import polars as pl
import pandas as pd


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

    orders_pd[
        ["order_delivered_customer_date", "order_purchase_timestamp"]
    ] = orders_pd[["order_delivered_customer_date", "order_purchase_timestamp"]].apply(
        pd.to_datetime
    )

    orders_pd["order_estimated_delivery_date"] = pd.to_datetime(
        orders_pd["order_estimated_delivery_date"], format="%Y-%m-%d %H:%M:%S"
    )

    orders_pd["label_actual_delivery_duration"] = (
        orders_pd["order_delivered_customer_date"]
        - orders_pd["order_purchase_timestamp"]
    ).apply(lambda x: x / pd.Timedelta(1, "h"))

    orders_pd["label_estimated_delivery_duration"] = (
        orders_pd["order_estimated_delivery_date"]
        - orders_pd["order_purchase_timestamp"]
    ).apply(lambda x: x / pd.Timedelta(1, "h"))

    return pl.from_pandas(orders_pd)


def create_final_dataset(
    pay_df,
    num_items_per_order_df,
    num_cat_per_order_df,
    customer_locat_df,
    num_sellers_per_order_df,
    num_cities_per_order_df,
    orderHolidayPurchaseCustomer,
    orderHolidayPurchaseSeller,
    delivery_duration_df,
):
    final_dataset = (
        (pay_df.lazy())
        .join(num_items_per_order_df.lazy(), how="left", on="order_id")
        .join(num_cat_per_order_df.lazy(), how="left", on="order_id")
        .join(customer_locat_df.lazy(), how="left", on="order_id")
        .join(num_sellers_per_order_df.lazy(), how="left", on="order_id")
        .join(num_cities_per_order_df.lazy(), how="left", on="order_id")
        .join(orderHolidayPurchaseCustomer.lazy(), how="left", on="order_id")
        .join(orderHolidayPurchaseSeller.lazy(), how="left", on="order_id")
        .join(delivery_duration_df.lazy(), how="left", on="order_id")
    )
    final_dataset = final_dataset.collect()
    final_dataset.write_csv("data/backup/exploratory_data_set.csv")

    return (
        final_dataset.drop(
            [
                "order_estimated_delivery_date",
                "order_delivered_customer_date",
                "customer_id",
                "order_id",
            ]
        )
        .drop_nulls()
        .to_pandas()
    )

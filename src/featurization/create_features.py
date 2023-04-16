import polars as pl

# TODO: Add docstrings
# TODO: Figure out how to import functions fro other modules
# TODO: Bring in brazilian holiday data prep
# TODO:  Check for missing import statements


def get_feature_total_payment_value(payments_df):
    return (
        payments_df.lazy()
        .select(pl.col(["order_id", "payment_value"]))
        .rename({"payment_value": "feat_total_payment_value"})
        .collect()
    )


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
    return (
        ord_items_df.lazy()
        .join(products_df.lazy(), how="left", on="product_id")
        .select(pl.col(["order_id", "order_item_id", "product_category_name"]))
        .select(
            [
                pl.col("order_id"),
                pl.col("product_category_name")
                .unique()
                .count()
                .over("order_id")
                .alias("feat_num_cat_per_order"),
            ]
        )
        .collect()
    )


def get_feature_customer_locat(orders_df, customers_df):
    return (
        orders_df.lazy()
        .select(pl.col(["order_id", "customer_id"]))
        .join(
            customers_df.lazy().select(
                pl.col(["customer_id", "customer_city", "customer_state"])
            ),
            how="left",
            on="customer_id",
        )
        .rename(
            {
                "customer_city": "feat_customer_city",
                "customer_state": "feat_customer_state",
            }
        )
        .collect()
    )


def get_feature_num_sellers(ord_items_df, sellers_df):
    return (
        ord_items_df.lazy()
        .join(sellers_df.lazy(), how="left", on="seller_id")
        .select(
            [
                pl.col("order_id"),
                pl.col("seller_id")
                .unique()
                .count()
                .over("order_id")
                .alias("feat_num_sellers_per_order"),
            ]
        )
        .collect()
    )


def get_feature_num_cities(ord_items_df, sellers_df):
    return (
        ord_items_df.lazy()
        .join(sellers_df.lazy(), how="left", on="seller_id")
        .select(
            [
                pl.col("order_id"),
                pl.col("seller_city")
                .unique()
                .count()
                .over("order_id")
                .alias("feat_num_seller_cities_per_order"),
            ]
        )
        .collect()
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


def get_feature_seller_order_holidays(orders_df, allHolidaysByOrderSeller):
    return (
        orders_df.select(pl.col("order_id"))
        .join(
            allHolidaysByOrderSeller.select(pl.col(["order_id", "isHoliday"])),
            how="left",
            on="order_id",
        )
        .groupby(by="order_id")
        .agg(
            [
                pl.col("isHoliday").sum().alias("feat_seller_holiday_impact"),
            ]
        )
        .with_columns(pl.col("feat_seller_holiday_impact").fill_null(0))
        .sort("feat_seller_holiday_impact", descending=True)
    )

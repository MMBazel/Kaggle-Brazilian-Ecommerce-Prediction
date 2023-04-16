import polars as pl

# TODO: Import SQL queries for the seller & customer scripts
# TODO: Check if there are missing import statements


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


def register_initial_SQL_Context(
    national_holidays_df, state_holidays_df, city_holidays_df
):
    sql = pl.SQLContext()
    sql.register("national_holidays_df", national_holidays_df.lazy())
    sql.register("state_holidays_df", state_holidays_df.lazy())
    sql.register("city_holidays_df", city_holidays_df.lazy())

    return sql


def get_customer_regions_by_order(orders_df, customers_df):
    return (
        orders_df.lazy()
        .select(pl.col(["order_id", "customer_id", "order_purchase_timestamp"]))
        .join(
            customers_df.lazy().select(
                pl.col(["customer_id", "customer_city", "customer_state"])
            ),
            how="left",
            on="customer_id",
        )
        .with_columns(
            pl.col("order_purchase_timestamp")
            .str.strptime(pl.Date, fmt="%Y-%m-%d %H:%M:%S")
            .cast(pl.Datetime)
        )
        .collect()
    )


def query_create_customer_holiday(sql, get_customer_regions_by_order_df):

    sql.register(
        "get_customer_regions_by_order", get_customer_regions_by_order_df.lazy()
    )

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


def get_seller_regions_by_order(orders_df, ord_items_df, sellers_df):
    return (
        orders_df.lazy()
        .select(pl.col(["order_id", "order_purchase_timestamp"]))
        .join(
            ord_items_df.lazy().select(
                pl.col(["order_id", "order_item_id", "seller_id"])
            ),
            how="left",
            on="order_id",
        )
        .join(
            sellers_df.lazy().select(
                pl.col(["seller_id", "seller_city", "seller_state"])
            ),
            how="left",
            on="seller_id",
        )
        .with_columns(
            pl.col("order_purchase_timestamp")
            .str.strptime(pl.Date, fmt="%Y-%m-%d %H:%M:%S")
            .cast(pl.Datetime)
        )
        .collect()
    )


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

    return pl.concat(
        [
            isNationalHolidaySellerOrder,
            isStateHolidaySellerOrder,
            isCityHolidaySellerOrder,
        ]
    )

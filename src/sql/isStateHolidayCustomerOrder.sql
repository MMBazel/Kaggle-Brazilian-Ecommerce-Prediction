-- TODO: Check this is the right format for a SQL file
 -- TODO: Write docstring
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
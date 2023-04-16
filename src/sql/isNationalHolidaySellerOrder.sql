-- TODO: Check this is the right format for a SQL file
 -- TODO: Write docstring
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
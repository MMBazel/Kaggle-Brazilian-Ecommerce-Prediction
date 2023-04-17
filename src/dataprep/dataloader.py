import polars as pl
import shutil


# TODO: Get values from config.cfg


def unzip_files(filename, target_dir):
    shutil.unpack_archive(filename, target_dir)


# TODO: Get rid of the hardcoded path name
def read_files(full_path, filename):
    df = pl.scan_csv(f"{full_path}/data/raw/{filename}").collect()

    return df


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

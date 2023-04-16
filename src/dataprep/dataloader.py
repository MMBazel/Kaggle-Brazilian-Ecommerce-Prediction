import polars as pl
import shutil


# TODO: Get values from config.cfg


def unzip_files(filename, target_dir):
    shutil.unpack_archive(filename, target_dir)


# TODO: Get rid of the hardcoded path name
def read_files(full_path, filename):
    df = pl.scan_csv(f"{full_path}/data/raw/{filename}").collect()

    return df

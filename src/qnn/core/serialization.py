import datetime
import yaml
import pandas as pd


def dict_to_yaml_file(filename: str, d: dict):
    with open(filename, 'w') as f:
        yaml.dump(d, f, default_flow_style=False)


def load_dict_from_yaml_file(filename: str):
    with open(filename, 'r') as f:
        d = yaml.safe_load(f)
    return d


def pandas_dataframe_to_csv_file(df: pd.DataFrame, output_filename: str):
    df.to_csv(output_filename, encoding='utf-8', decimal='.')


def load_pandas_dataframe_from_csv_file(filename: str, dtype=None) -> pd.DataFrame:
    return pd.read_csv(filename, encoding='utf-8', decimal='.', parse_dates=['date'], dtype=dtype, date_parser=lambda x: datetime.datetime.utcfromtimestamp(int(x)), index_col='date')

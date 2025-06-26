import os
from typing import Optional

import numpy as np
import pandas as pd
from syntherela.data import load_tables

from relbench.base import Database, Dataset, Table
from syntherela.metadata import Metadata

def get_tables_and_metadata(
    dataset: str, method: str, run_id: int
) -> tuple[dict[str:pd.DataFrame], Metadata]:
    data_type = "original" if method == "ORIGINAL" else "synthetic"
    path = os.path.join("data", data_type, dataset)
    if method != "ORIGINAL":
        path = os.path.join(path, method, str(run_id), "sample1")

    metadata_path = os.path.join("data", "original", dataset, "metadata.json")
    metadata = Metadata.load_from_json(metadata_path)

    tables = load_tables(path, metadata)

    return tables, metadata


class F1SyntheticDataset(Dataset):
    name = "f1_subsampled"
    val_timestamp = pd.Timestamp("2005-01-01")
    test_timestamp = pd.Timestamp("2010-01-01")

    # from_timestamp = pd.Timestamp("1990-01-01")
    # upto_timestamp = pd.Timestamp("2010-01-01")

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        predict_column_task_config: dict = {},
        method: str = "RelDiff",
        run_id: int = 1,
        type: str = "train",
    ):
        super().__init__(cache_dir)
        self.method = method
        self.run_id = run_id
        self.type = type

    def make_db(self) -> Database:
        tables_train, metadata = get_tables_and_metadata(
            self.name, self.method, self.run_id
        )
        tables_test = load_tables(os.path.join("data", "original", "f1"), metadata)

        if self.type == "test":
            tables = tables_test
        else:
            tables = tables_train

        # Convert categorical columns to string
        for table_name, table in tables.items():
            categorical_cols = metadata.get_column_names(table_name, sdtype="categorical")
            for col in categorical_cols:
                if col in table.columns:
                    tables[table_name][col] = tables[table_name][col].astype(str)

        circuits = tables["circuits"]
        drivers = tables["drivers"]
        results = tables["results"]
        races = tables["races"]
        standings = tables["standings"]
        constructors = tables["constructors"]
        constructor_results = tables["constructor_results"]
        constructor_standings = tables["constructor_standings"]
        qualifying = tables["qualifying"]

        # Add date column to races by extracting date from datetime
        # races["date"] = pd.to_datetime(pd.to_datetime(races["datetime"]).dt.date)
        # races["time"] = pd.to_datetime(races["datetime"]).dt.time

        races.pop("year")
        races["date"] = pd.to_datetime(races.pop("datetime"))

        qualifying = qualifying.merge(
            races[["raceId", "date"]], on="raceId", how="left"
        )

        # # Subtract a day from the date to account for the fact
        # # that the qualifying time is the day before the main race
        qualifying["date"] = qualifying["date"] - pd.Timedelta(days=1)

        # Replace "\N" with NaN in results tables
        results = results.replace(r"^\\N$", np.nan, regex=True)

        # Replace "\N" with NaN in circuits tables, especially
        # for the column `alt` which has 3 rows of "\N"
        circuits = circuits.replace(r"^\\N$", np.nan, regex=True)
        # Convert alt from string to float
        circuits["alt"] = circuits["alt"].astype(float)

        # Convert non-numeric values to NaN in the specified column
        results["rank"] = pd.to_numeric(results["rank"], errors="coerce")
        results["number"] = pd.to_numeric(results["number"], errors="coerce")
        results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
        results["position"] = pd.to_numeric(results["position"], errors="coerce")
        results["points"] = pd.to_numeric(results["points"], errors="coerce")
        results["laps"] = pd.to_numeric(results["laps"], errors="coerce")
        results["milliseconds"] = pd.to_numeric(
            results["milliseconds"], errors="coerce"
        )
        results["fastestLap"] = pd.to_numeric(results["fastestLap"], errors="coerce")

        # Convert drivers date of birth to datetime
        drivers["dob"] = pd.to_datetime(drivers["dob"])

        tables = {}

        tables["races"] = Table(
            df=pd.DataFrame(races),
            fkey_col_to_pkey_table={
                "circuitId": "circuits",
            },
            pkey_col="raceId",
            time_col="date",
        )

        tables["circuits"] = Table(
            df=pd.DataFrame(circuits),
            fkey_col_to_pkey_table={},
            pkey_col="circuitId",
            time_col=None,
        )

        tables["drivers"] = Table(
            df=pd.DataFrame(drivers),
            fkey_col_to_pkey_table={},
            pkey_col="driverId",
            time_col=None,
        )

        tables["results"] = Table(
            df=pd.DataFrame(results),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers",
                "constructorId": "constructors",
            },
            pkey_col="resultId",
            time_col="date",
        )

        tables["standings"] = Table(
            df=pd.DataFrame(standings),
            fkey_col_to_pkey_table={"raceId": "races", "driverId": "drivers"},
            pkey_col="driverStandingsId",
            time_col="date",
        )

        tables["constructors"] = Table(
            df=pd.DataFrame(constructors),
            fkey_col_to_pkey_table={},
            pkey_col="constructorId",
            time_col=None,
        )

        tables["constructor_results"] = Table(
            df=pd.DataFrame(constructor_results),
            fkey_col_to_pkey_table={"raceId": "races", "constructorId": "constructors"},
            pkey_col="constructorResultsId",
            time_col="date",
        )

        tables["constructor_standings"] = Table(
            df=pd.DataFrame(constructor_standings),
            fkey_col_to_pkey_table={"raceId": "races", "constructorId": "constructors"},
            pkey_col="constructorStandingsId",
            time_col="date",
        )

        tables["qualifying"] = Table(
            df=pd.DataFrame(qualifying),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers",
                "constructorId": "constructors",
            },
            pkey_col="qualifyId",
            time_col="date",
        )

        db = Database(tables)

        # db = db.from_(self.from_timestamp)
        # db = db.upto(self.upto_timestamp)

        return db
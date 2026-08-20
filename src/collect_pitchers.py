from pathlib import Path
import pandas as pd

from pybaseball import pitching_stats


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SEASONS = [2022, 2023, 2024]

OUTPUT_FILE = Path(
    "data/raw/mlb_pitching_stats.csv"
)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("========================================")
    print("MLB STARTING PITCHER DATA COLLECTION")
    print("========================================")

    all_data = []

    for season in SEASONS:

        print("")
        print(
            f"Downloading pitching statistics "
            f"for {season}..."
        )

        try:

            data = pitching_stats(
                season,
                season,
                qual=0
            )

            if data is None or data.empty:

                print(
                    f"ERROR: No pitching data for "
                    f"{season}"
                )

                continue

            data = data.copy()

            data["Season"] = season

            print(
                f"Pitchers collected: "
                f"{len(data)}"
            )

            print(
                "Columns available:"
            )

            print(
                data.columns.tolist()
            )

            all_data.append(data)

        except Exception as error:

            print(
                f"ERROR downloading {season}: "
                f"{error}"
            )

    if not all_data:

        print("")
        print(
            "ERROR: No pitching data collected."
        )

        return

    # -----------------------------------------------------
    # COMBINE
    # -----------------------------------------------------

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("")
    print("========================================")
    print("PITCHING DATA COLLECTION COMPLETE")
    print("========================================")

    print(
        f"Total pitcher rows: "
        f"{len(combined)}"
    )

    print(
        f"Seasons: {SEASONS}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    # -----------------------------------------------------
    # IMPORTANT COLUMNS
    # -----------------------------------------------------

    print("")
    print(
        "Important pitching columns:"
    )

    important = [

        "Name",
        "Team",
        "ERA",
        "WHIP",
        "SO",
        "BB",
        "IP",
        "W",
        "L"
    ]

    available = [
        column
        for column in important
        if column in combined.columns
    ]

    for column in available:

        print(
            f"  - {column}"
        )

    # -----------------------------------------------------
    # SAMPLE
    # -----------------------------------------------------

    print("")
    print(
        "First 10 pitchers:"
    )

    columns = [
        column
        for column in [
            "Season",
            "Name",
            "Team",
            "ERA",
            "WHIP",
            "SO",
            "BB",
            "IP",
            "W",
            "L"
        ]
        if column in combined.columns
    ]

    print(
        combined[columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":

    main()
from pathlib import Path

import pandas as pd

"""
Generate data_paths arg file in the following format:
"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"
"""

TRAIN_START_END = "0:0.969"
VALID_START_END = "0.969:0.999"
TEST_START_END = "0.999:1"

# DATA_PATH = "/fsx/loubna/data/tokenized_stack_no_pii"

# $DATA_PATH: variable to be replaced later with the directory containing the tokenized data
# The $DATA_PATH directory should contain `code`, `gh_commits`, `gh_issues`, `jupyter_scripts/structured` subdirectories
DATA_ENV = r"${DATA_PATH}"
OTHER_SOURCES_PATHS = {
    # "gh issues": f"{DATA_ENV}/",  # TODO
    "gh commits": f"{DATA_ENV}/gh_commits/gpt2-preprocessed_content_document",
    "notebook scripts": f"{DATA_ENV}/jupyter_scripts/gpt2-preprocessed_content_document",
    "notebook structured": f"{DATA_ENV}/jupyter_structured/gpt2-preprocessed_content_document",
    "gh issues": f"{DATA_ENV}/gh_issues/gpt2-preprocessed_content_document",
}

EXCLUDE_FROM_TEST = [
    "bluespec",
    "verilog",
    "matlab",
    "augeas"
]

def main():
    data_sources_path = "data/data_sources.csv"
    out_path = Path("data")

    data_sources = pd.read_csv(data_sources_path)

    data_sources["data_prefix"] = data_sources["Data-source"].apply(
        lambda lang: OTHER_SOURCES_PATHS.get(lang, f"{DATA_ENV}/code/{lang}/gpt2-preprocessed_content_document")
    )

    print(data_sources)
    # save in a file
    data_sources.to_csv("res_data_sources.csv", index=False)
    # Training: only 1 group with all data sources.
    train_data_group = ", ".join([
        f"{row['Weight']} {TRAIN_START_END} {row['data_prefix']}" for i, row in data_sources.iterrows()
    ])
    train_data_arg = f"\"TRAIN: {train_data_group}\"\n"
    with open(out_path / "train_data_paths.txt", 'w') as f:
        f.write(train_data_arg)

    # Validation and test
    def get_grouped_args(split_arg: str, split_name, exclude_sources=None):
        if exclude_sources is None:
            exclude_sources = []
        # 1 group for each source, and the global group with weights
        global_data_group = ", ".join([
            f"{row['Weight']} {split_arg} {row['data_prefix']}"
            for i, row in data_sources.iterrows() if row['Data-source'] not in exclude_sources
        ])
        global_data_arg = f"\"{split_name}_all_sources_weighted: {global_data_group}\""

        data_args = [
            f"\"{split_name}_{row['Data-source'].replace(' ', '_')}: 1 {split_arg} {row['data_prefix']}\""
            for i, row in data_sources.iterrows() if row["Data-source"] not in exclude_sources
        ]
        return ' '.join(data_args + [global_data_arg]) + "\n"
    
    valid_data_arg = get_grouped_args(VALID_START_END, "VALID")
    with open(out_path / "valid_data_paths.txt", 'w') as f:
        f.write(valid_data_arg)
        
    test_data_arg = get_grouped_args(TEST_START_END, "TEST", exclude_sources=EXCLUDE_FROM_TEST)
    with open(out_path / "test_data_paths.txt", 'w') as f:
        f.write(test_data_arg)
    

if __name__ == "__main__":
    main()

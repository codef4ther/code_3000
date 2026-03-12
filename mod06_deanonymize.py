import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    # Identify quasi-identifier columns (shared between both DataFrames, excluding ID/name cols)
    exclude_cols = {"anon_id", "name", "id"}
    anon_cols = set(anon_df.columns)
    aux_cols = set(aux_df.columns)
    quasi_identifiers = list((anon_cols & aux_cols) - exclude_cols)

    if not quasi_identifiers:
        raise ValueError("No shared quasi-identifier columns found between datasets.")

    # Merge on quasi-identifiers
    merged = anon_df.merge(aux_df, on=quasi_identifiers, how="inner")

    # Keep only the anon_id and matched name
    merged = merged[["anon_id", "name"]].rename(columns={"name": "matched_name"})

    # Retain only UNIQUE matches (one auxiliary record per anonymized record)
    unique_matches = merged.drop_duplicates(subset=["anon_id"], keep=False)

    return unique_matches.reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    total_records = len(anon_df)
    if total_records == 0:
        return 0.0
    
    unique_reidentified = len(matches_df)
    return unique_reidentified / total_records
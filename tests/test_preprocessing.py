"""Activité 2 — tests unitaires du contrat de données et du prétraitement."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import StandardScaler

from app import config
from app.preprocessing import (
    ENGINEERED_COLUMNS,
    DataValidationError,
    load_dataset,
    preprocess_data,
    split_data,
    validate_data,
)
from tests.conftest import make_synthetic

# --------------------------------------------------------------------------- T01


def test_valid_data_passes_with_all_required_columns(synthetic_df):
    out, report = validate_data(synthetic_df)
    assert list(out.columns) == config.REQUIRED_COLUMNS
    assert report.n_rows == len(synthetic_df)
    assert report.warnings == []


@pytest.mark.parametrize("column", ["Time", "V1", "V14", "V28", "Amount", "Class"])
def test_missing_required_column_is_rejected(synthetic_df, column):
    with pytest.raises(DataValidationError, match=column):
        validate_data(synthetic_df.drop(columns=[column]))


def test_target_not_required_for_scoring(synthetic_df):
    out, _ = validate_data(synthetic_df.drop(columns=["Class"]), require_target=False)
    assert "Class" not in out.columns


def test_duplicated_column_is_rejected(synthetic_df):
    dup = pd.concat([synthetic_df, synthetic_df[["Amount"]]], axis=1)
    with pytest.raises(DataValidationError, match="dupliquées.*Amount"):
        validate_data(dup)


def test_extra_column_ignored_in_batch_but_rejected_in_strict(synthetic_df):
    df = synthetic_df.assign(customer_email="x@y.z")
    out, report = validate_data(df)
    assert "customer_email" not in out.columns
    assert report.dropped_extra_columns == ["customer_email"]
    with pytest.raises(DataValidationError, match="non autorisées"):
        validate_data(df, strict=True)


# --------------------------------------------------------------------------- T02


@pytest.mark.parametrize("bad_value", ["abc", None, True])
def test_non_numeric_feature_is_rejected(synthetic_df, bad_value):
    df = synthetic_df.copy()
    df["V3"] = df["V3"].astype(object)
    df.loc[0, "V3"] = bad_value
    if bad_value is None:  # None dans une colonne object reste non numérique
        df["V3"] = df["V3"].astype(object)
    with pytest.raises(DataValidationError, match="V3"):
        validate_data(df)


def test_boolean_column_is_rejected(synthetic_df):
    df = synthetic_df.assign(Amount=synthetic_df["Amount"] > 10)
    with pytest.raises(DataValidationError, match="Amount"):
        validate_data(df)


def test_integer_columns_are_accepted_and_cast_to_float(synthetic_df):
    df = synthetic_df.assign(Time=synthetic_df["Time"].astype(int))
    out, _ = validate_data(df)
    assert out["Time"].dtype == np.float64


@pytest.mark.parametrize("bad_target", [2, -1, 0.5])
def test_target_outside_0_1_is_rejected(synthetic_df, bad_target):
    df = synthetic_df.copy()
    df["Class"] = df["Class"].astype(float)
    df.loc[3, "Class"] = bad_target
    with pytest.raises(DataValidationError, match="cible"):
        validate_data(df)


def test_missing_target_is_rejected(synthetic_df):
    df = synthetic_df.copy()
    df["Class"] = df["Class"].astype(float)
    df.loc[0, "Class"] = np.nan
    with pytest.raises(DataValidationError, match="cible"):
        validate_data(df)


def test_negative_amount_is_rejected(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Amount"] = -1.0
    with pytest.raises(DataValidationError, match="négatif"):
        validate_data(df)


def test_absurd_amount_is_rejected_but_limit_is_accepted(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Amount"] = config.AMOUNT_MAX_REJECT
    validate_data(df)  # la borne elle-même est acceptée
    df.loc[0, "Amount"] = config.AMOUNT_MAX_REJECT + 0.01
    with pytest.raises(DataValidationError, match="borne"):
        validate_data(df)


def test_negative_time_is_rejected(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Time"] = -5
    with pytest.raises(DataValidationError, match="Time"):
        validate_data(df)


# --------------------------------------------------------------------- vides / 1 ligne


def test_empty_dataframe_is_rejected():
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(pd.DataFrame(columns=config.REQUIRED_COLUMNS))


def test_non_dataframe_input_is_rejected():
    with pytest.raises(DataValidationError, match="DataFrame"):
        validate_data([[1, 2, 3]])


def test_single_observation_is_validated_and_preprocessed(synthetic_df):
    one = synthetic_df.iloc[[0]]
    out, _ = validate_data(one)
    feats = preprocess_data(out)
    assert feats.shape == (1, len(ENGINEERED_COLUMNS))
    assert np.isfinite(feats.to_numpy()).all()


# ------------------------------------------------------------- NaN / inf (T03)


def test_missing_values_are_reported_in_batch_and_rejected_in_strict(synthetic_df):
    df = synthetic_df.copy()
    df.loc[[0, 1], "V2"] = np.nan
    _, report = validate_data(df)
    assert report.n_missing_values == 2
    assert any("imputées" in w for w in report.warnings)
    with pytest.raises(DataValidationError, match="strict"):
        validate_data(df, strict=True)


def test_infinite_values_are_reported_and_converted_to_nan(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "V7"] = np.inf
    df.loc[1, "V8"] = -np.inf
    _, report = validate_data(df)
    assert report.n_infinite_values == 2
    feats = preprocess_data(df)
    assert not np.isinf(feats.to_numpy()).any()
    assert feats[["V7", "V8"]].isna().sum().sum() == 2  # imputées ensuite par le pipeline
    with pytest.raises(DataValidationError):
        validate_data(df, strict=True)


def test_pipeline_imputation_yields_finite_output(model, X_test):
    X = X_test.head(50).copy()
    X.iloc[0, 3] = np.nan
    X.iloc[1, 5] = np.inf
    transformed = model.pipeline[:-1].transform(X)
    assert np.isfinite(transformed).all()


# --------------------------------------------------------------- transformation


def test_preprocess_dimensions_and_columns(synthetic_df):
    feats = preprocess_data(synthetic_df)
    assert list(feats.columns) == ENGINEERED_COLUMNS
    assert feats.shape == (len(synthetic_df), 31)
    assert feats.index.equals(synthetic_df.index)


def test_preprocess_is_stateless_and_deterministic(synthetic_df):
    a = preprocess_data(synthetic_df)
    b = preprocess_data(synthetic_df.iloc[::-1]).loc[a.index]
    pd.testing.assert_frame_equal(a, b)


def test_log_amount_and_cyclic_hour_values():
    df = make_synthetic(3)
    df["Amount"] = [0.0, np.e - 1, 1_000_000.0]
    df["Time"] = [0.0, 6 * 3600.0, 86_400.0 + 0.0]  # minuit, 6h, minuit J+1
    feats = preprocess_data(df)
    np.testing.assert_allclose(feats["log_amount"][:2], [0.0, 1.0])
    np.testing.assert_allclose(feats["hour_sin"], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(feats["hour_cos"], [1.0, 0.0, 1.0], atol=1e-12)


def test_preprocess_rejects_missing_column(synthetic_df):
    with pytest.raises(DataValidationError):
        preprocess_data(synthetic_df.drop(columns=["V5"]))


@settings(max_examples=60, deadline=None)
@given(amount=st.floats(0, 1e6), time=st.floats(0, 1e7))
def test_preprocess_output_is_finite_for_any_valid_amount_and_time(amount, time):
    df = make_synthetic(1)
    df["Amount"], df["Time"] = amount, time
    feats = preprocess_data(df)
    assert np.isfinite(feats.to_numpy()).all()
    assert -1 <= feats["hour_sin"].iloc[0] <= 1


# ------------------------------------------------------------------- exemple 2.2


def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


def test_scaler_constant_column_does_not_produce_nan():
    """Valeur limite : variance nulle -> sklearn renvoie 0, pas NaN."""
    scaled = StandardScaler().fit_transform(pd.DataFrame({"Amount": [5.0, 5.0, 5.0]}))
    assert np.isfinite(scaled).all()


# ---------------------------------------------------------------- split / chargement


def test_split_is_stratified_disjoint_and_deduplicated(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.head(20)])  # 20 doublons exacts
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df)
    assert len(X_tr) + len(X_va) + len(X_te) == len(synthetic_df)
    assert not (set(X_tr.index) & set(X_te.index))
    assert not (set(X_tr.index) & set(X_va.index))
    rate = synthetic_df["Class"].mean()
    for y in (y_tr, y_va, y_te):
        assert abs(y.mean() - rate) < 0.03


def test_split_is_reproducible_with_seed(synthetic_df):
    a = split_data(synthetic_df, seed=7)
    b = split_data(synthetic_df, seed=7)
    c = split_data(synthetic_df, seed=8)
    assert a[2].index.equals(b[2].index)
    assert not a[2].index.equals(c[2].index)


def test_load_dataset_errors_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.csv")
    bad_ext = tmp_path / "data.xlsx"
    bad_ext.write_text("x")
    with pytest.raises(DataValidationError, match="Extension"):
        load_dataset(bad_ext)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataValidationError, match="illisible"):
        load_dataset(empty)
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n")
    with pytest.raises(DataValidationError, match="absentes"):
        load_dataset(wrong)

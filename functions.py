import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from math import ceil
import scipy.stats as stats
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.base import BaseEstimator, TransformerMixin


def Categorical_to_binary(df, value1, value0):
    """Replace "yes" with 1 and "no" with 0"""
    df1 = df.applymap(lambda x: 1 if x == value1 else (0 if x == value0 else x))
    return df1


def unique_values(df):
    """Lists all unique values in columns"""
    unique_values_dict = {}

    for column in df.columns:
        unique_values_dict[column] = df[column].unique().tolist()
    return unique_values_dict


# PLOTTING


def my_countplot(df, feature1, feature2, title):
    """Plots the count plot of df, setting feature1 as name of X axis,
    feature2 as name of 'hue' parameter"""

    ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=feature1, hue=feature2, data=df)
    ax.set_title(title)
    ax.set(ylabel="")
    ax.set(xlabel=feature1)
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])


def my_proportionsplot(df, feature1, feature2, title):
    """Plots the proportion plot of df, setting feature1 as name of X axis,
    feature2 as name of 'hue' parameter"""

    sns.histplot(
        data=df,
        x=feature1,
        hue=feature2,
        multiple="fill",
        stat="proportion",
        discrete=True,
        shrink=0.8,
    ).set(title=title)


def my_plots(
    df: pd.DataFrame, feature1: str, feature2: str, title1: str, title2: str
) -> None:
    """Plots count plot and proportion plots of categorical features per target variable.
    params: df: usable pd.DataFrame;
    feature1: str - name of the categorical column, which values to plot;
    feature2: str - name of the target feature;"""
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(x=feature1, hue=feature2, data=df, ax=ax[0])
    ax[0].set_title(title1)
    sns.histplot(
        data=df,
        x=feature1,
        hue=feature2,
        multiple="fill",
        stat="proportion",
        discrete=True,
        shrink=0.8,
        ax=ax[1],
    )
    ax[1].set_title(title2)
    plt.close(2)
    plt.show()


def plot_kde(df: pd.DataFrame, feature1: str, feature2: str) -> None:
    """Plots KDE plot distribution of numerical feature with categorical "hue".
    params: df: usable pd.DataFrame;
            feature1: str - name of the numerical column, which values to plot;
            feature2: str - name of the categorical binary (1 or 0) column used as a "hue";
    """
    plt.figure(figsize=(15, 5))
    plt.title(f"KDE Plot: {feature1} vs. {feature2}", fontsize=30, fontweight="bold")
    ax = sns.kdeplot(
        df[df[feature2] == 1][feature1],
        color="red",
        label=f"had {feature2}",
        lw=2,
        legend=True,
    )
    ax1 = sns.kdeplot(
        df[df[feature2] == 0][feature1],
        color="blue",
        label=f"no {feature2}",
        lw=2,
        legend=True,
    )
    legend = ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    sns.despine(right=True, left=True)
    plt.tight_layout()


# STATISTICAL INFERENCE


def Power_test_one_tail(proportion1: float, proportion2: float) -> None:
    """Prints the needed sample size to avoid p-hacking for one tail test.
    params: proportion1: float - proportion of positive target values in first group of interest;
            proportion2: float - proportion of positive target values in second group of interest;
    """
    effect_size = sms.proportion_effectsize(proportion1, proportion2)
    required_n = sms.NormalIndPower().solve_power(
        effect_size, power=0.8, alpha=0.05, ratio=1
    )
    required_n = ceil(required_n)
    print(f" Required sample size:{required_n}")


def calc_pi_t_test_proportions(df_emp: pd.DataFrame):
    """Calculates pi value using t test for difference in proportions
    params: df_emp: DataFrame
    preparation of df:
        df_emp = df.groupby("feature of interest")[["target value"]].agg(["sum", "count"])
        df_emp = df_emp.droplevel(0, axis=1).reset_index()
        df_emp["proportion"] = df_emp["sum"] / df_emp["count"]
        df_emp"""
    # Calculation of Standard error of estimate:
    p_both = (df_emp.iloc[0]["sum"] + df_emp.iloc[1]["sum"]) / (
        df_emp.iloc[0]["count"] + df_emp.iloc[1]["count"]
    )  # Common sample proportion
    va = p_both * (1 - p_both)
    se = np.sqrt(va * (1 / df_emp.iloc[0]["count"] + 1 / df_emp.iloc[1]["count"]))
    # Calculate the t test statistic
    test_stat = (df_emp.iloc[0]["proportion"] - df_emp.iloc[1]["proportion"]) / se
    pvalue = stats.norm.sf(abs(test_stat))
    print(f"Pi value for diff in proportions using t test:{pvalue}")


def calc_confid_intervals(df_emp: pd.DataFrame) -> None:
    """Calculates confidens intervals for difference in proportions
        using Confidence level of 95%, significant level alpha = 0.05
    params: df_emp: DataFrame
    preparation of df:
        df_emp = df.groupby("feature of interest")[["target value"]].agg(["sum", "count"])
        df_emp = df_emp.droplevel(0, axis=1).reset_index()
        df_emp["proportion"] = df_emp["sum"] / df_emp["count"]
        df_emp"""
    # SE0
    p0 = df_emp.iloc[0]["proportion"]
    n0 = df_emp.iloc[0]["sum"]  # Total number of purchases
    st_err0 = p0 * (1 - p0) / n0

    # SE1
    p1 = df_emp.iloc[1]["proportion"]
    n1 = df_emp.iloc[1]["sum"]  # Total number of purchases
    st_err1 = p1 * (1 - p1) / n1

    # sqrt(SE0+SE1) = Standar error
    se = np.sqrt(st_err0 + st_err1)

    # First interval value
    intv_1 = (p0 - p1) - 1.96 * se
    # Second interval value
    intv_0 = (p0 - p1) + 1.96 * se
    print(f"Confidence interval is   {intv_1.round(2)} - {intv_0.round(2)}")


# MODELING


def confusion_matrix_normalized(
    y_val: pd.DataFrame, pred_y: pd.DataFrame, labels: list
) -> None:
    """Plots normalized confusion matrix.
    :param: y_val: pd.DataFrame with features;
            pred_y: pd.DataFrame dependent variable;
            labels: matrix labels
    """
    cm = confusion_matrix(y_val, pred_y)
    # Normalise
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show(block=False)


# Feature engineering


def get_bmi_times_glucose(df):
    df["bmi_times_glucose"] = df["bmi"] * df["avg_glucose_level"]
    return df.drop(["bmi", "avg_glucose_level"], axis=1)


def get_age2_per_bmi(df):
    df["glucose_per_age"] = df["age"] * df["age"] / df["bmi"]
    return df.drop(["age", "bmi"], axis=1)

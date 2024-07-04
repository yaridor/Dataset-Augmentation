import pandas as pd
from tabulate import tabulate
import numpy as np
from scipy.io import arff
from matplotlib import pyplot as plt

from typing import Callable


def read_arff(data_path: str):
    # Reading ARFF file
    data, meta = arff.loadarff(data_path)

    data = pd.DataFrame(data, columns=meta.names())
    data = data.apply(pd.to_numeric)

    return data


def add_samples(data: pd.DataFrame, num_samples: int, augmentation_factor: int, noise_level: dict[str, float]):
    """
    Augment new samples

    :param data: The data to add samples to
    :param num_samples: The number of samples to base the augmentation on (negative for all)
    :param augmentation_factor: The factor by which to augment the data
    :param noise_level: The level of noise for the augmentation for each feature
    :return: The data with the added samples. Output size = num_samples * (augmentation_factor - 1) + len(data)
    """
    if num_samples < 0:
        num_samples = len(data)

    # Select samples to augment
    base_sample_indices = np.random.choice(len(data), num_samples)
    base_sample_indices = np.tile(base_sample_indices, augmentation_factor)
    base_samples = data.iloc[base_sample_indices]

    # Augment samples
    augmented_samples = pd.DataFrame()
    for column in data.columns:
        columns_type = data[column].dtype
        if column in noise_level:
            noise = np.random.normal(0, noise_level[column], (num_samples * augmentation_factor,))
            augmented_samples[column] = noise + base_samples[column].values
            augmented_samples[column] = augmented_samples[column].astype(columns_type)
        else:
            augmented_samples[column] = base_samples[column].values

    # Remove original samples
    data = data.drop(base_sample_indices)

    # Add augmented samples
    data = pd.concat([data, augmented_samples])

    return data


def apply_restriction(data: pd.DataFrame, column: str, restriction_function: Callable):
    """
    Apply a restriction to the data

    :param data: The data to apply the restriction to
    :param column: The column to apply the restriction to
    :param restriction_function: The restriction function to apply (e.g. lambda x: x if x > 0 else 0)
    :return: The data with the restriction applied
    """

    data[column] = data[column].apply(restriction_function)
    return data


def add_feature(data: pd.DataFrame, base_features: list[str], new_feature_name: str, feature_function: Callable):
    """
    Add a new feature to the data

    :param data: The data to add the feature to
    :param base_features: The features to base the new feature on
    :param new_feature_name: The name of the new feature
    :param feature_function: The function to calculate the new feature
    :return: The data with the new feature added
    """

    data[new_feature_name] = data[base_features].apply(feature_function, axis=1)
    return data


def conditional_sample(mean: float, std: float, condition: Callable):
    sample = np.random.normal(mean, std)
    while not condition(sample):
        sample = np.random.normal(mean, std)

    return sample


def main():
    data_path = "caesarian.csv.arff"
    data = read_arff(data_path)

    # Restore possible values for each feature
    delivery_time = {
        0: lambda: int(conditional_sample(40, 5, lambda x: 38 <= x < 43)),  # Timely
        1: lambda: int(conditional_sample(40, 5, lambda x: 0 <= x < 39)),  # Premature
        2: lambda: int(conditional_sample(40, 5, lambda x: 43 <= x < 90)),  # Latecomer
    }
    data = apply_restriction(data, "Delivery time", lambda x: delivery_time[x]())
    data.rename(columns={"Delivery time": "Delivery Week"}, inplace=True)

    systolic_blood_pressure = {
        0: lambda: int(conditional_sample(120, 50, lambda x: 40 <= x < 90)),  # Low
        1: lambda: int(conditional_sample(120, 50, lambda x: 90 <= x < 140)),  # Normal
        2: lambda: int(conditional_sample(120, 50, lambda x: 140 <= x < 200)),  # High
    }
    diastolic_blood_pressure = {
        0: lambda: int(conditional_sample(75, 30, lambda x: 20 <= x < 60)),  # Low
        1: lambda: int(conditional_sample(75, 30, lambda x: 60 <= x < 90)),  # Normal
        2: lambda: int(conditional_sample(75, 30, lambda x: 90 <= x < 140)),  # High
    }
    data = add_feature(data, ["Blood Pressure"], "Systolic Blood Pressure",
                       lambda x: systolic_blood_pressure[x["Blood Pressure"]]())
    data = add_feature(data, ["Blood Pressure"], "Diastolic Blood Pressure",
                       lambda x: diastolic_blood_pressure[x["Blood Pressure"]]())
    data.drop("Blood Pressure", axis=1, inplace=True)

    data["Heart Problem"] = data["Heart Problem"].astype(bool)

    # Add samples
    data = add_samples(data, -1, 501,
                       {"Age": 2,
                        "Delivery Week": 3,
                        "Systolic Blood Pressure": 20,
                        "Diastolic Blood Pressure": 8})

    # Add flipping noise to heart problem
    data["Heart Problem"] = data["Heart Problem"].apply(lambda x: not x if np.random.rand() < 0.1 else x)
    heart_problem_examples = {
        False: ["No Heart Problem"],
        True: ["Coronary Artery Disease",
               "Heart Arrhythmias",
               "Heart Failure",
               "Heart Valve Disease",
               "Pericardial Disease",
               "Cardiomyopathy",
               "Congenital Heart Disease",
               ],
    }
    data = apply_restriction(data, "Heart Problem", lambda x: np.random.choice(heart_problem_examples[x]))

    # Apply restrictions for each feature
    # Anonymize sensitive age data
    data = apply_restriction(data, "Age", lambda x: x if x > 20 else None)
    # Anonymize sensitive delivery week data
    data = apply_restriction(data, "Delivery Week", lambda x: x if 20 < x < 60 else None)
    data["Systolic Blood Pressure"] = data["Systolic Blood Pressure"].clip(30, 220)
    data["Diastolic Blood Pressure"] = data["Diastolic Blood Pressure"].clip(20, 160)

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # Print data examples
    print(tabulate(data.head(50), headers="keys", tablefmt="psql"))
    print("Num samples:", len(data))
    print("Num features:", len(data.columns))

    # Save data
    data.to_csv("data.csv", index=False)


if __name__ == '__main__':
    main()

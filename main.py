import pandas as pd
from tabulate import tabulate
import numpy as np

from scipy.io import arff
from ucimlrepo import fetch_ucirepo

import numbers
from typing import Callable, Union, Optional, Any


# <editor-fold desc="I/O">

def read_arff(data_path: str):
    # Reading ARFF file
    data, meta = arff.loadarff(data_path)

    data = pd.DataFrame(data, columns=meta.names())
    data = data.apply(pd.to_numeric)

    return data


def read_from_uci(data_id: int):
    data = fetch_ucirepo(id=data_id)
    x, y = data.data.features, data.data.targets
    variables_description = data.variables
    return pd.concat([x, y], axis=1), variables_description


# </editor-fold>

# <editor-fold desc="Sample Augmentation">
def add_samples(data: pd.DataFrame, num_samples: int, augmentation_factor: int,
                noise_levels: dict[str, Union[float, np.ndarray]]):
    """
    Augment new samples

    :param data: The data to add samples to
    :param num_samples: The number of samples to base the augmentation on (negative for all)
    :param augmentation_factor: The factor by which to augment the data
    :param noise_levels: The level of noise for the augmentation for each feature
    :return: The data with the added samples. Output size = num_samples * (augmentation_factor - 1) + len(data)
    """
    if num_samples < 0:
        num_samples = len(data)

    # Select samples to augment
    base_sample_indices = np.random.choice(len(data), num_samples, replace=False)
    base_sample_indices = np.tile(base_sample_indices, augmentation_factor)
    base_samples = data.iloc[base_sample_indices]

    # Augment samples
    augmented_samples = pd.DataFrame()
    for column in data.columns:
        columns_type = data[column].dtype
        if column in noise_levels:
            if isinstance(noise_levels[column], numbers.Number):
                # Add Gaussian noise
                noise = np.random.normal(0, noise_levels[column], (num_samples * augmentation_factor,))
                augmented_samples[column] = noise + base_samples[column].values
            elif isinstance(noise_levels[column], np.ndarray):
                # Random flips according to the flipping matrix
                options = list(range(data[column].max().item() + 1))
                weights = noise_levels[column] / noise_levels[column].sum(axis=1)
                noisy_values = base_samples[column].apply(lambda x: np.random.choice(options, p=weights[x]))
                augmented_samples[column] = noisy_values.values
            augmented_samples[column] = augmented_samples[column].astype(columns_type)
        else:
            augmented_samples[column] = base_samples[column].values

    # Remove original samples
    data = data.drop(base_sample_indices)

    # Add augmented samples
    data = pd.concat([data, augmented_samples])

    return data


# </editor-fold>

# <editor-fold desc="Feature Augmentation">

def apply_transformation(data: pd.DataFrame, column: str, transformation: Callable):
    """
    Apply a restriction to the data

    :param data: The data to apply the restriction to
    :param column: The column to apply the restriction to
    :param transformation: The restriction function to apply (e.g. lambda x: x if x > 0 else 0)
    :return: The data with the restriction applied
    """

    data[column] = data[column].apply(transformation)
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


def conditional_sample(distribution: Callable, condition: Callable, *args, **kwargs):
    """
    Draws a sample from a distribution with a given arguments, while satisfying a condition

    :param distribution: The distribution to sample from
    :param condition: The condition to satisfy
    :return:
    """
    sample = distribution(*args, **kwargs)
    while not condition(sample):
        sample = distribution(*args, **kwargs)

    return sample


def conditional_range_sample(distribution: Callable, min_value: float, max_value: float, *args, **kwargs):
    """
    Draws a sample from a distribution with a given arguments, while satisfying a condition

    :param distribution: The distribution to sample from
    :param min_value: The minimum value to satisfy
    :param max_value: The maximum value to satisfy
    :return:
    """
    return conditional_sample(distribution, lambda x: min_value <= x < max_value, *args, **kwargs)


# </editor-fold>

# <editor-fold desc="Missing Data">

def add_missing_data(data: pd.DataFrame, column: str, missing_rate: float,
                     condition: Optional[Callable] = None, empty_value: Any = None):
    """
    Remove some data to simulate missing data

    :param data: The dataframe to use
    :param column: The column for which missing data should be added
    :param missing_rate: The rate of missing data to add (0 <= missing_rate <= 1)
    :param condition: Missing data will only be added to rows for which this condition is satisfied
    :param empty_value: The value to replace the data with
    :return:
    """
    assert 0 <= missing_rate <= 1

    if condition is None:
        condition = lambda x: True

    data[column] = data[column].apply(lambda x: empty_value if condition(x) and np.random.rand() < missing_rate else x)

    return data


# </editor-fold>

def main():
    data, description = read_from_uci(887)
    data.rename(columns={"RIAGENDR": "Gender", "PAQ605": "Sport Activity", "BMXBMI": "BMI", "LBXGLU": "Glucose",
                         "DIQ010": "Diabetic", "LBXGLT": "Oral", "LBXIN": "Insulin Level",
                         "age_group": "Age Group"}, inplace=True)
    print(tabulate(data.head(50), headers="keys", tablefmt="psql"))

    data_path = "Original Data/caesarian.csv.arff"
    data = read_arff(data_path)

    data.to_csv("Output Data/caesarian_original.csv", index=False)

    # Restore possible values for each feature
    delivery_time = {
        0: lambda: int(conditional_range_sample(np.random.normal, 39, 43, loc=40, scale=5)),  # Timely
        1: lambda: int(conditional_range_sample(np.random.normal, 0, 39, loc=40, scale=5)),  # Timely
        2: lambda: int(conditional_range_sample(np.random.normal, 43, 90, loc=40, scale=5)),  # Timely
    }
    data = apply_transformation(data, "Delivery time", lambda x: delivery_time[x]())
    data.rename(columns={"Delivery time": "Delivery Week",
                         "Delivery number": "Delivery Number"},
                inplace=True)

    # Add samples with noise
    data = add_samples(data, -1, 501, {
        "Age": 2,
        "Delivery Week": 3,
        "Blood Pressure": np.array([[0.8, 0.1, 0.1],
                                    [0.2, 0.7, 0.1],
                                    [0.3, 0.2, 0.5]]),
        "Heart Problem": np.array([[0.9, 0.1],
                                   [0.1, 0.9]]),
    })

    # Adding features
    systolic_blood_pressure = {
        0: lambda: int(conditional_range_sample(np.random.normal, 40, 90, loc=120, scale=50, )),
        1: lambda: int(conditional_range_sample(np.random.normal, 90, 140, loc=120, scale=50, )),
        2: lambda: int(conditional_range_sample(np.random.normal, 140, 200, loc=120, scale=50, )),
    }
    diastolic_blood_pressure = {
        0: lambda: int(conditional_range_sample(np.random.normal, 20, 60, loc=75, scale=30, )),
        1: lambda: int(conditional_range_sample(np.random.normal, 60, 90, loc=75, scale=30, )),
        2: lambda: int(conditional_range_sample(np.random.normal, 90, 140, loc=75, scale=30, )),
    }
    data = add_feature(data, ["Blood Pressure"], "Systolic Blood Pressure",
                       lambda x: systolic_blood_pressure[x["Blood Pressure"]]())
    data["Systolic Blood Pressure"] = data["Systolic Blood Pressure"].clip(30, 220)
    data = add_feature(data, ["Blood Pressure"], "Diastolic Blood Pressure",
                       lambda x: diastolic_blood_pressure[x["Blood Pressure"]]())
    data["Diastolic Blood Pressure"] = data["Diastolic Blood Pressure"].clip(20, 160)

    data = add_feature(data, ["Heart Problem", "Blood Pressure"], "Medical Family History",
                       lambda x: "Yes" if np.random.randint(0, 4) +
                                          x["Heart Problem"] * np.random.randint(1, 8) +
                                          abs(x["Blood Pressure"] - 1) * np.random.randint(2, 7) > 5
                       else "No")

    nationalities = ["American", "British", "Chinese", "Indian", "Japanese", "Korean", "Russian", "Turkish"]
    weights = np.array([8, 1, 2, 4, 1, 0.1, 3, 2])
    data = add_feature(data, ["Age"], "Nationality",
                       lambda x: np.random.choice(nationalities, p=weights / weights.sum()))

    data = add_feature(data, ["Caesarian"], "Academic Level",
                       lambda x: int(np.random.normal(7 - 3 * x["Caesarian"], 1)))
    data["Academic Level"] = data["Academic Level"].clip(1, 9)

    data = add_feature(data, ["Caesarian"], "Testing Feature",
                       lambda x: np.random.normal(7 - 3 * x["Caesarian"], 1))

    # Undo binary heart problem categorization
    heart_problem_options = {
        0: ["No Heart Problem"],
        1: ["Coronary Artery Disease",
            "Heart Arrhythmias",
            "Heart Failure",
            "Heart Valve Disease",
            "Pericardial Disease",
            "Cardiomyopathy",
            "Congenital Heart Disease",
            ],
    }
    weights = {
        0: np.array([1]),
        1: np.array([3, 1, 4, 1, 5, 9, 2])
    }
    data = apply_transformation(data, "Heart Problem",
                                lambda x: np.random.choice(heart_problem_options[x], p=weights[x] / weights[x].sum()))

    # Remove unnecessary features
    data.drop("Blood Pressure", axis=1, inplace=True)

    # Apply restrictions and missing data
    data = add_missing_data(data, "Age", 1, lambda x: x < 18)
    data = add_missing_data(data, "Delivery Week", 1, lambda x: x < 30 or x > 45)
    data = add_missing_data(data, "Systolic Blood Pressure", 0.1, lambda x: 80 <= x < 100)
    data = add_missing_data(data, "Systolic Blood Pressure", 0.4, lambda x: 50 <= x < 70)
    data = add_missing_data(data, "Diastolic Blood Pressure", 0.15)
    data = add_missing_data(data, "Academic Level", 0.3, empty_value=10)  # label 10 is missing data
    data = add_missing_data(data, "Delivery Week", 0.05)
    data = add_missing_data(data, "Nationality", 0.2)
    data = add_missing_data(data, "Medical Family History", 0.1)
    data = add_missing_data(data, "Delivery Number", 0.3)
    data = add_missing_data(data, "Heart Problem", 0.12)

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # Print data examples
    print(tabulate(data.head(50), headers="keys", tablefmt="psql"))
    print("Num samples:", len(data))
    print("Num features:", len(data.columns))

    # Save data
    data.to_csv("Output Data/caesarian.csv", index=False)


if __name__ == "__main__":
    main()

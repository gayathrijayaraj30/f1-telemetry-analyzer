import pandas as pd
import numpy as np
import fastf1
from fastf1.core import Laps
from sklearn.preprocessing import StandardScaler

def extract_driver_features(laps: Laps) -> pd.DataFrame:
    year = pd.to_datetime(laps.session.event['EventDate']).year
    gp = laps.session.event['EventName']
    session_name = laps.session.name
    session = fastf1.get_session(year, gp, session_name)
    session.load()

    features = []
    for driver in laps['Driver'].unique():
        driver_laps = laps.pick_driver(driver)
        if driver_laps.empty:
            continue
        fastest_lap = driver_laps.pick_fastest()
        if fastest_lap is None:
            continue

        try:
            car_data = fastest_lap.get_car_data().add_distance()
        except Exception:
            continue

        try:
            info = session.get_driver(driver)
        except Exception:
            continue

        rpm_diff = car_data['RPM'].diff().abs()

        features.append({
            "driver": driver,
            "driver_number": info.DriverNumber,
            "abbreviation": info.Abbreviation,
            "full_name": info.FullName,
            "brake_mean": car_data['Brake'].astype(float).mean(),  # FIXED
            "throttle_mean": car_data['Throttle'].mean(),          # FIXED
            "speed_mean": car_data['Speed'].mean(),
            "speed_std": car_data['Speed'].std(),
            "rpm_mean": car_data['RPM'].mean(),
            "max_rpm_accel": rpm_diff.rolling(window=5, min_periods=1).max().max(),
            "gear_changes": car_data['nGear'].diff().abs().fillna(0).astype(bool).sum(),
            "avg_gear": car_data['nGear'].mean(),
            "throttle_std": car_data['Throttle'].std(),
            "brake_std": car_data['Brake'].std(),
            "sharp_brakes": ((car_data['Brake'] > 0.7) & (car_data['Speed'] > 50)).sum(),
            "total_distance": car_data['Distance'].max()
        })

    return pd.DataFrame(features)


def scale_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    scaled = StandardScaler().fit_transform(df[feature_cols])
    return pd.DataFrame(scaled, columns=feature_cols, index=df.index)
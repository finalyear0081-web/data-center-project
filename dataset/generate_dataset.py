"""
Synthetic Data Center Temperature Dataset Generator
Produces a high-fidelity, physically consistent 1-year simulated sensor dataset (105,120 records).
Sampling interval: 5 minutes.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_data_center_dataset(output_path="dataset/data_center_temperature_dataset.csv"):
    np.random.seed(42)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    total_steps = 105120  # 365 days * 24 hours * 12 (5-min intervals)
    
    # Generate timestamp sequence
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(total_steps)]
    
    # Time factors
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps])
    hour_of_day = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps]) # 0=Monday, 6=Sunday
    is_weekend = (day_of_week >= 5).astype(float)
    
    # 1. Outdoor Temperature (°C): Seasonal (winter cold, summer warm) + Diurnal (day peak ~15:00) + noise
    seasonal_outdoor = 18.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    diurnal_outdoor = 6.0 * np.sin(2 * np.pi * (hour_of_day - 9.0) / 24.0)
    outdoor_noise = np.random.normal(0, 1.2, total_steps)
    outdoor_temperature = np.clip(seasonal_outdoor + diurnal_outdoor + outdoor_noise, -5.0, 42.0)
    
    # 2. CPU Usage (%): High during business hours (09:00 - 18:00 on weekdays), lower at night and weekends
    workload_base = 55.0 - 20.0 * is_weekend
    workload_diurnal = 25.0 * np.sin(np.maximum(0, (hour_of_day - 6.0)) * np.pi / 14.0)
    workload_diurnal = np.where((hour_of_day >= 7) & (hour_of_day <= 20), workload_diurnal, 5.0)
    workload_noise = np.random.normal(0, 4.5, total_steps)
    cpu_usage = np.clip(workload_base + workload_diurnal + workload_noise, 10.0, 98.0)
    
    # 3. IT Power Consumption (kW): Proportional to CPU usage + static base load (servers, network switches, storage)
    # Range typically 220 kW to 460 kW
    it_power_consumption = 200.0 + (cpu_usage / 100.0) * 220.0 + np.random.normal(0, 3.5, total_steps)
    
    # 4. Cooling Power Consumption (kW): Baseline chiller + fan power + active cooling load
    # Higher during hot days and peak compute
    cooling_power_base = 70.0 + (it_power_consumption - 200.0) * 0.22 + (outdoor_temperature - 15.0) * 0.8
    cooling_power_consumption = np.clip(cooling_power_base + np.random.normal(0, 4.0, total_steps), 45.0, 175.0)
    
    # 5. Cold Aisle Temperature (°C): Target setpoint ~ 20.0°C with small fluctuations
    cold_aisle_temperature = 20.0 + 0.015 * (outdoor_temperature - 20.0) + 0.01 * (it_power_consumption - 300.0) - 0.02 * (cooling_power_consumption - 90.0) + np.random.normal(0, 0.4, total_steps)
    cold_aisle_temperature = np.clip(cold_aisle_temperature, 17.5, 23.5)
    
    # 6. Server Inlet Temperature (°C): Target variable!
    # Server inlet draws air from cold aisle, but thermal mixing, server density, and rack elevation
    # cause it to vary between ~21°C and 29.5°C.
    # Excursions above 28°C occur periodically during summer afternoons + peak compute workloads.
    summer_peak = np.maximum(0.0, (day_of_year - 150) / 40.0) * np.maximum(0.0, (250 - day_of_year) / 40.0)
    inlet_temperature = (
        cold_aisle_temperature 
        + 2.2 
        + 0.015 * (it_power_consumption - 200.0) 
        + 0.08 * np.maximum(0.0, outdoor_temperature - 22.0)
        + 1.8 * np.sin(2 * np.pi * (hour_of_day - 13.5) / 24.0)
        + 1.2 * (summer_peak > 0.5).astype(float)
        + np.random.normal(0, 0.35, total_steps)
    )
    # Ensure natural excursions cross 28.0°C and dip below 26.0°C for realistic cooling hysteresis
    inlet_temperature = np.clip(inlet_temperature, 19.5, 30.5)
    
    # 7. Server Outlet Temperature (°C): Air heated across servers (Delta T ~ 10-15°C)
    delta_t = 8.5 + (cpu_usage / 100.0) * 8.0 + np.random.normal(0, 0.5, total_steps)
    outlet_temperature = inlet_temperature + delta_t
    
    # 8. Hot Aisle Temperature (°C): Hot exhaust air collected before return plenum
    hot_aisle_temperature = outlet_temperature + 1.2 + np.random.normal(0, 0.4, total_steps)
    
    # 9. Humidity (%): Relative humidity maintained around 45% - 60%
    humidity = 52.0 - 0.4 * (inlet_temperature - 22.0) - 0.2 * (outdoor_temperature - 18.0) + np.random.normal(0, 1.8, total_steps)
    humidity = np.clip(humidity, 35.0, 70.0)
    
    df = pd.DataFrame({
        'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        'inlet_temperature': np.round(inlet_temperature, 2),
        'outlet_temperature': np.round(outlet_temperature, 2),
        'hot_aisle_temperature': np.round(hot_aisle_temperature, 2),
        'cold_aisle_temperature': np.round(cold_aisle_temperature, 2),
        'humidity': np.round(humidity, 2),
        'cpu_usage': np.round(cpu_usage, 2),
        'it_power_consumption': np.round(it_power_consumption, 2),
        'outdoor_temperature': np.round(outdoor_temperature, 2),
        'cooling_power_consumption': np.round(cooling_power_consumption, 2)
    })
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at '{output_path}'.")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Timestamp range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print("\nSummary Statistics for inlet_temperature:")
    print(df['inlet_temperature'].describe())
    return df

if __name__ == '__main__':
    generate_data_center_dataset()
    

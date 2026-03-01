import pandas as pd
import numpy as np
import os

def generate_realistic_agronomy_data(n_samples=50000):
    np.random.seed(42)
    
    crops = ['Wheat', 'Barley', 'Cotton', 'Maize', 'Rice', 'Soybean']
    soil_types = ['Sandy', 'Loamy', 'Clay', 'Peaty', 'Chalky', 'Silty']
    
    # Biometric profiles (Optimal Temp Range, Optimal Rain Range, Base Yield Potential)
    crop_profiles = {
        'Wheat': {'opt_temp': (15, 25), 'opt_rain': (400, 600), 'base_yield': 4.5, 'opt_days': 100},
        'Barley': {'opt_temp': (12, 22), 'opt_rain': (300, 500), 'base_yield': 4.0, 'opt_days': 90},
        'Cotton': {'opt_temp': (25, 35), 'opt_rain': (600, 1000), 'base_yield': 3.0, 'opt_days': 160},
        'Maize': {'opt_temp': (18, 27), 'opt_rain': (500, 800), 'base_yield': 6.0, 'opt_days': 120},
        'Rice': {'opt_temp': (20, 30), 'opt_rain': (1000, 1500), 'base_yield': 5.0, 'opt_days': 130},
        'Soybean': {'opt_temp': (20, 30), 'opt_rain': (500, 900), 'base_yield': 3.5, 'opt_days': 110}
    }
    
    # Soil multiplier (Loamy is generally best, Sandy worst for water retention)
    soil_multipliers = {
        'Loamy': 1.2, 'Clay': 1.0, 'Silty': 1.1, 'Chalky': 0.9, 'Peaty': 0.8, 'Sandy': 0.7
    }

    data = {
        'Soil_Type': np.random.choice(soil_types, n_samples),
        'Crop': np.random.choice(crops, n_samples),
        'Rainfall_mm': np.random.uniform(0, 2000, n_samples),
        'Temperature_Celsius': np.random.uniform(-10, 60, n_samples),
        'Fertilizer_Used': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
        'Irrigation_Used': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
        'Days_to_Harvest': np.random.randint(30, 300, n_samples)
    }
    
    df = pd.DataFrame(data)
    yields = []
    
    for i, row in df.iterrows():
        c_prof = crop_profiles[row['Crop']]
        
        # Base
        current_yield = c_prof['base_yield']
        
        # Soil Modifier
        current_yield *= soil_multipliers[row['Soil_Type']]
        
        # Temperature Penalty (Bell curve drop off)
        t_opt_min, t_opt_max = c_prof['opt_temp']
        if row['Temperature_Celsius'] < t_opt_min:
            penalty = (t_opt_min - row['Temperature_Celsius']) / 10.0
            current_yield *= max(0.1, 1.0 - penalty) # Drop heavily if freezing
        elif row['Temperature_Celsius'] > t_opt_max:
            penalty = (row['Temperature_Celsius'] - t_opt_max) / 10.0
            current_yield *= max(0.1, 1.0 - penalty) # Drop heavily if burning
            
        # Rainfall Penalty
        r_opt_min, r_opt_max = c_prof['opt_rain']
        if row['Rainfall_mm'] < r_opt_min:
            # Drought
            penalty = (r_opt_min - row['Rainfall_mm']) / 300.0
            if not row['Irrigation_Used']:
                current_yield *= max(0.1, 1.0 - penalty)
            else:
                # Irrigation saves it partially
                current_yield *= max(0.5, 1.0 - (penalty * 0.3))
        elif row['Rainfall_mm'] > r_opt_max:
            # Flood
            penalty = (row['Rainfall_mm'] - r_opt_max) / 500.0
            current_yield *= max(0.1, 1.0 - penalty)
            
        # Harvest Days Penalty (premature or rotting)
        h_diff = abs(row['Days_to_Harvest'] - c_prof['opt_days'])
        if h_diff > 20:
            current_yield *= max(0.2, 1.0 - (h_diff / 100.0))
            
        # Fertilizer Bonus
        if row['Fertilizer_Used']:
            current_yield *= 1.3
            
        # Add slight natural noise (±10%)
        noise = np.random.uniform(0.9, 1.1)
        current_yield *= noise
        
        # Final safety bounds
        yields.append(round(max(0.0, current_yield), 2))
        
    df['Yield_tons_per_hectare'] = yields
    
    # Save it over existing dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(BASE_DIR, "data", "crop_yield.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {n_samples} rows of strictly correlated Agronomy Data at {out_path}!")

if __name__ == "__main__":
    generate_realistic_agronomy_data()

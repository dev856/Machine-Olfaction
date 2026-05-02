"""Create demo sensor CSV files for testing the Streamlit app."""
import numpy as np
import pandas as pd
from pathlib import Path

def create_sample_csv(output_path, smell_class, n_points=200):
    """Create a realistic-looking sensor response CSV."""
    np.random.seed(42)
    
    # Time column
    time = np.linspace(0, 100, n_points)
    
    # Simulate sensor responses with different patterns per smell class
    sensors = ['TGS2600', 'TGS2602', 'TGS2610', 'TGS2611', 'TGS2620']
    
    data = {'time': time}
    
    # Base parameters vary by smell class
    base_params = {
        'coffee': {'peak': 0.8, 'decay': 0.03, 'noise': 0.02},
        'banana': {'peak': 0.6, 'decay': 0.04, 'noise': 0.03},
        'lemon': {'peak': 0.7, 'decay': 0.05, 'noise': 0.025},
        'lavender': {'peak': 0.5, 'decay': 0.02, 'noise': 0.015},
        'vanilla': {'peak': 0.65, 'decay': 0.025, 'noise': 0.02},
    }
    
    params = base_params.get(smell_class, base_params['coffee'])
    
    for i, sensor in enumerate(sensors):
        # Create response curve with rise and decay
        peak_time = 20 + i * 5
        response = params['peak'] * np.exp(-0.5 * ((time - peak_time) / 10) ** 2)
        response += params['peak'] * 0.3 * np.exp(-params['decay'] * time)
        response += np.random.normal(0, params['noise'], n_points)
        response = np.clip(response, 0, 1.5)
        data[sensor] = response
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path}")

# Create output directory
output_dir = Path("/workspace/data/samples")
output_dir.mkdir(parents=True, exist_ok=True)

# Create samples for different smell classes
smell_classes = ['coffee', 'banana', 'lemon', 'lavender', 'vanilla']

for smell_class in smell_classes:
    output_path = output_dir / f"{smell_class}_trial_001.csv"
    create_sample_csv(output_path, smell_class)

print(f"\nCreated {len(smell_classes)} demo sample files in {output_dir}")

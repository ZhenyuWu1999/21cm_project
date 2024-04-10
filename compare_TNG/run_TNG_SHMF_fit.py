import subprocess
import numpy as np
# List of snapshot numbers
snapshots = np.arange(5, 14, 1)
snapshots = np.append(snapshots, np.arange(15, 27, 2))
snapshots = np.append(snapshots, np.array([29,33,40,50,67]))
print(snapshots)

# Loop over the snapshot numbers
for snapshot in snapshots:
    # Run the script with the snapshot number as an argument
    subprocess.run(["python", "read_TNG.py", str(snapshot)])
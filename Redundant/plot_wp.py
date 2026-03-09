import os
import csv
import matplotlib.pyplot as plt
# cwd = os.getcwd()  # Get the current working directory (cwd)

# Read from the CSV file
with open('./utils/lists.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)

    # Initialize empty lists
    ego_wp_x, ego_wp_y, oppo_wp_x, oppo_wp_y = [], [], [], []

    for row in reader:
        ego_wp_x.append(float(row[0]))
        ego_wp_y.append(float(row[1]))
        oppo_wp_x.append(float(row[2]))
        oppo_wp_y.append(float(row[3]))

lane_width = 4.0

fig = plt.figure()
ax = fig.subplots()
ax.plot(ego_wp_y, ego_wp_x, ".", color="b")
ax.plot(oppo_wp_y, oppo_wp_x, ".", color="r")
plt.show()
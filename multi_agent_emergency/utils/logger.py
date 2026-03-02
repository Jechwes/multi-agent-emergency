import numpy as np
import csv

def trajectory_record(ego_fea_extract, opp_fea_extract):
    ego_wp_x, ego_wp_y, oppo_wp_x, oppo_wp_y = [], [], [], []
    for wp in ego_fea_extract.all_wp:
        ego_wp_x.append(wp.x)
        ego_wp_y.append(wp.y)
    for wp in opp_fea_extract.all_wp:
        oppo_wp_x.append(wp.x)
        oppo_wp_y.append(wp.y)

    # Zip the lists to write them row-wise
    rows = zip(ego_wp_x, ego_wp_y, oppo_wp_x, oppo_wp_y)

    # Write to a CSV file
    with open('lists.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ego_wp_x', 'ego_wp_y', 'oppo_wp_x', 'oppo_wp_y'])  # Header
        writer.writerows(rows)


def policy_record(occ_measure, joint_state):
    occ_measure = np.array(occ_measure)
    np.savetxt('offline_policy/ply_'+ str(joint_state) +'.csv', occ_measure, delimiter=',')

def get_offline_policy(joint_state):
    try:
        occ_measure = np.loadtxt('offline_policy/ply_'+ str(joint_state) +'.csv', delimiter=',')
        return occ_measure
    except:
        return None

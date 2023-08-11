import os

from imu_filter.parse_data import main

if __name__ == "__main__":
    graph_name = os.environ.get("graph_name", "orientation_vs_estimate.png")
    main(orientation_estimates_path_name=graph_name)
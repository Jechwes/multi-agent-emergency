"""
visualize_roundabout_lanelets.py
================================
Spawn a CARLA world (Town03) and draw the roundabout lanelet map.

Run:
    python visualize_roundabout_lanelets.py [--host 127.0.0.1] [-p 2000]

The script:
1. Connects to a running CARLA server and loads Town03 (the roundabout map).
2. Moves the spectator camera directly above the roundabout.
3. Builds a ``RoundaboutLaneletMap`` with the configured geometry.
4. Draws all section boundaries, lane centre-lines, lane edges, and
   longitudinal grid lines within each section using CARLA debug lines.
5. Keeps the simulation ticking so the visualisation stays visible.

Press Ctrl+C to exit.
"""

import argparse
import math
import time
import sys
import os

import numpy as np

# Ensure sibling package imports work when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from abstraction.roundabout_lanelets import RoundaboutLaneletMap

import carla


def main():
    # ------------------------------------------------------------------ CLI
    parser = argparse.ArgumentParser(description="Visualise roundabout lanelets in CARLA")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("-p", "--port", default=2000, type=int, help="CARLA server port")
    parser.add_argument("--life-time", default=60.0, type=float,
                        help="Debug line lifetime [s]  (-1 = persistent)")
    parser.add_argument("--s-grid", default=None, type=float,
                        help="Longitudinal grid spacing within each section [m]. "
                             "If omitted, no transverse grid lines are drawn.")
    args = parser.parse_args()

    # --------------------------------------------------------- CARLA connect
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.load_world("Town03")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.tick()

    # ------------------------------------------------ Spectator (top-down)
    spectator = world.get_spectator()
    # Town03 roundabout is near (x=-0.5, y=0.5) — look straight down
    spectator.set_transform(carla.Transform(
        carla.Location(x=-0.5, y=0.5, z=70),
        carla.Rotation(pitch=-90),
    ))

    # ----------------------------------------- Roundabout lanelet geometry
    #  Match the parameters you use in main_roundabout.py / check_grid.py
    CENTRE       = (-0.5, 0.5)
    INNER_RADIUS = 13.5     # [m] inner edge of ring 0
    LANE_WIDTH   = 4.0      # [m] radial width per lane
    N_LANES      = 4        # number of concentric rings
    N_SECTIONS   = 24       # number of equal-angle sectors

    rmap = RoundaboutLaneletMap(
        centre=CENTRE,
        inner_radius=INNER_RADIUS,
        lane_width=LANE_WIDTH,
        n_lanes=N_LANES,
        n_sections=N_SECTIONS,
        direction="ccw",
    )

    print(rmap.summary())
    print()

    # ---------------------------------------------------------- Draw map
    print("Drawing lanelet map in CARLA …")
    rmap.draw_in_carla(
        world,
        z=0.25,
        n_samples=120,
        s_grid_step=args.s_grid,
        life_time=args.life_time,
        draw_section_boundaries=True,
        draw_lane_boundaries=True,
        draw_centrelines=True,
        draw_s_grid=(args.s_grid is not None),
        draw_section_labels=True,
    )
    world.tick()

    # Also print per-section arc-lengths so they can be verified
    print("\nPer-section arc lengths:")
    for lane_id in range(N_LANES):
        L = rmap.section_arc_length(lane_id)
        print(f"  Lane {lane_id}: L_section = {L:.3f} m")

    # ------------------------------------------------ Keep world ticking
    print("\nVisualisation active. Press Ctrl+C to exit.\n")
    try:
        while True:
            world.tick()
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        # Restore async mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()

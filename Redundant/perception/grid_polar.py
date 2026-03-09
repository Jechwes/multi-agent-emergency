import carla
import math


class Griding():

    def __init__(self, world, origin_point, lane_start, lane_amount, cell_width, angle_step=5):
        self.world = world
        self.lane_start = lane_start
        self.origin_point = origin_point
        self.cell_width = cell_width
        self.lane_amount = lane_amount
        self.angle_step = angle_step

    def draw_grid_map_polar(self, time=-1.0):
        origin = self.origin_point
        edges = []

        radii = [self.lane_start + i * self.cell_width for i in range(self.lane_amount + 1)]
        
        angles_deg = []
        curr_angle = 0
        while curr_angle < 360:
            angles_deg.append(curr_angle)
            curr_angle += self.angle_step

        angles_rad = [math.radians(a) for a in angles_deg]
        
        for r in radii:
            circle_points = []
            for theta in angles_rad:
                # Calculate x and y relative to origin without changing z
                x = origin.x + r * math.cos(theta)
                y = origin.y + r * math.sin(theta)
                circle_points.append(carla.Location(x=x, y=y, z=origin.z))
            
            # Close the loop ONLY if 360 is not in angles_deg (range excludes stop value)
            # If 360 % angle_step == 0, the last point connects to the first
            circle_points.append(circle_points[0]) 

            for k in range(len(circle_points) - 1):
                edges.append((circle_points[k], circle_points[k+1]))
        
        # 2. Draw Radial Lines
        r_min = radii[0]
        r_max = radii[-1]
        
        for theta in angles_rad:
            p_start = carla.Location(
                x=origin.x + r_min * math.cos(theta),
                y=origin.y + r_min * math.sin(theta),
                z=origin.z
            )
            p_end = carla.Location(
                x=origin.x + r_max * math.cos(theta),
                y=origin.y + r_max * math.sin(theta),
                z=origin.z
            )
            edges.append((p_start, p_end))

        # Draw all collected edges
        for edge in edges:
            self.world.debug.draw_line(
                edge[0],
                edge[1],
                thickness=0.06,
                color=carla.Color(r=0, g=255, b=0, a=100),  # Green color similar to old method
                life_time=time,
                persistent_lines=True)

    def draw_box(self, grid_range, time=-1):
        ring_idx = grid_range[0]
        sector_idx = grid_range[1]
        
        # Calculate angular range
        theta_start_deg = sector_idx * self.angle_step
        theta_end_deg = (sector_idx + grid_range[3]) * self.angle_step
        
        # Calculate radial range
        r_inner = self.lane_start + ring_idx * self.cell_width
        r_outer = r_inner + grid_range[2] * self.cell_width
        
        edges = []
        
        # Arc at r_inner
        steps = max(1, int((theta_end_deg - theta_start_deg) / 5)) # every 5 degrees or at least 1 step
        arc_angles = [math.radians(theta_start_deg + i * (theta_end_deg - theta_start_deg)/steps) for i in range(steps + 1)]

        for i in range(len(arc_angles)-1):
            p1 = carla.Location(
                self.origin_point.x + r_inner * math.cos(arc_angles[i]),
                self.origin_point.y + r_inner * math.sin(arc_angles[i]),
                self.origin_point.z + 0.1  # slightly above ground
            )
            p2 = carla.Location(
                self.origin_point.x + r_inner * math.cos(arc_angles[i+1]),
                self.origin_point.y + r_inner * math.sin(arc_angles[i+1]),
                self.origin_point.z + 0.1 
            )
            edges.append((p1, p2))
            
            p3 = carla.Location(
                self.origin_point.x + r_outer * math.cos(arc_angles[i]),
                self.origin_point.y + r_outer * math.sin(arc_angles[i]),
                self.origin_point.z + 0.1
            )
            p4 = carla.Location(
                self.origin_point.x + r_outer * math.cos(arc_angles[i+1]),
                self.origin_point.y + r_outer * math.sin(arc_angles[i+1]),
                self.origin_point.z + 0.1
            )
            edges.append((p3, p4))
            
        # Radial lines closing the sector
        p_start_inner = carla.Location(
            self.origin_point.x + r_inner * math.cos(arc_angles[0]),
            self.origin_point.y + r_inner * math.sin(arc_angles[0]),
            self.origin_point.z + 0.1
        )
        p_start_outer = carla.Location(
            self.origin_point.x + r_outer * math.cos(arc_angles[0]),
            self.origin_point.y + r_outer * math.sin(arc_angles[0]),
            self.origin_point.z + 0.1
        )
        edges.append((p_start_inner, p_start_outer))
        
        p_end_inner = carla.Location(
            self.origin_point.x + r_inner * math.cos(arc_angles[-1]),
            self.origin_point.y + r_inner * math.sin(arc_angles[-1]),
            self.origin_point.z + 0.1
        )
        p_end_outer = carla.Location(
            self.origin_point.x + r_outer * math.cos(arc_angles[-1]),
            self.origin_point.y + r_outer * math.sin(arc_angles[-1]),
            self.origin_point.z + 0.1
        )
        edges.append((p_end_inner, p_end_outer))

        for edge in edges:
            self.world.debug.draw_line(
                edge[0],
                edge[1],
                thickness=0.1,  # Thicker than grid lines
                color=carla.Color(r=255, g=0, b=0),  # Red color
                life_time=time,
                persistent_lines=True
            )


    def draw_point(self, point):
        self.world.debug.draw_point(
            carla.Location(point[0], point[1], 0.2),
            size=0.2,
            color=carla.Color(r=30, g=0, b=0),
            life_time=0.2
        )

    def draw_path(self, path):
        for n in range(1, len(path)):
             # Path node is (ring_idx, sector_idx, heading_unused)
             r_idx_from, s_idx_from = path[n-1][0], path[n-1][1]
             r_idx_to, s_idx_to = path[n][0], path[n][1]
             
             r_mid_from = self.lane_start + (r_idx_from + 0.5) * self.cell_width
             theta_mid_from = math.radians((s_idx_from + 0.5) * self.angle_step)
             
             r_mid_to = self.lane_start + (r_idx_to + 0.5) * self.cell_width
             theta_mid_to = math.radians((s_idx_to + 0.5) * self.angle_step)

             point_from = carla.Location(
                 x=self.origin_point.x + r_mid_from * math.cos(theta_mid_from),
                 y=self.origin_point.y + r_mid_from * math.sin(theta_mid_from),
                 z=self.origin_point.z + 0.2
             )
             
             point_to = carla.Location(
                 x=self.origin_point.x + r_mid_to * math.cos(theta_mid_to),
                 y=self.origin_point.y + r_mid_to * math.sin(theta_mid_to),
                 z=self.origin_point.z + 0.2
             )

             self.world.debug.draw_point(
                point_from,
                size=0.1,
                color=carla.Color(r=30, g=0, b=0),
                life_time=0.12)

             self.world.debug.draw_point(
                point_to,
                size=0.1,
                color=carla.Color(r=30, g=0, b=0),
                life_time=0.12)

             self.world.debug.draw_line(
                point_from,
                point_to,
                thickness=0.06,
                color=carla.Color(r=30, g=0, b=0, a=100),
                life_time=0.12,
                persistent_lines=True)


    def is_in_grid(self, pos, grid_range):
        grid_state = self.get_grid_state(pos)
        if ((grid_range[0] <= grid_state[0] < grid_range[0] + grid_range[2])
                and (grid_range[1] <= grid_state[1] < grid_range[1] + grid_range[3])):
            return True
        else:
            return False

    def get_grid_state(self, pos):
        dx = pos.x - self.origin_point.x
        dy = pos.y - self.origin_point.y
        
        # Calculate radius and angle
        r = math.sqrt(dx*dx + dy*dy)
        theta_rad = math.atan2(dy, dx)
        theta_deg = math.degrees(theta_rad)
        if theta_deg < 0:
            theta_deg += 360
            
        # Calculate indices
        ring_idx = int((r - self.lane_start) // self.cell_width)
        sector_idx = int(theta_deg // self.angle_step)
        
        return (ring_idx, sector_idx)


    def bbox_display(self, vehicle, color, time):
        if color == "red":
            _color = carla.Color(10, 0, 0, 0)
        elif color == "green":
            _color = carla.Color(0, 10, 0, 0)
        elif color == "blue":
            _color = carla.Color(0, 0, 10, 0)
        # Bounding Box
        bounding_box = vehicle.bounding_box
        bounding_box.location = vehicle.get_transform().location
        self.world.debug.draw_box(bounding_box, vehicle.get_transform().rotation,
                                  color=_color, thickness=0.1, life_time=time)






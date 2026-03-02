import carla


class Griding():

    def __init__(self, world, origin_point, grid_shape, cell_size):
        self.world = world
        self.origin_point = origin_point
        self.cell_size = cell_size
        self.grid_shape = grid_shape

    def draw_grid_map(self, time=-1.0):
        origin = self.origin_point
        edges = []

        for i in range(self.grid_shape[0] + 1):
            point_from = carla.Location(
                origin.x + i * self.cell_size[0],
                origin.y,
                origin.z)
            point_to = carla.Location(
                origin.x + i * self.cell_size[0],
                origin.y + self.grid_shape[1] * self.cell_size[1],
                origin.z)
            edges.append((point_from, point_to))

        for j in range(self.grid_shape[1] + 1):
            point_from = carla.Location(
                origin.x,
                origin.y + j * self.cell_size[1],
                origin.z)
            point_to = carla.Location(
                origin.x + self.grid_shape[0] * self.cell_size[0],
                origin.y + j * self.cell_size[1],
                origin.z)
            edges.append((point_from, point_to))

        for edge in edges:
            self.world.debug.draw_line(
                edge[0],
                edge[1],
                thickness=0.06,
                color=carla.Color(r=0, g=10, b=0, a=100),
                life_time=time,
                persistent_lines=True)



    def draw_box(self, grid_range):

        cell_size = self.cell_size
        box_loc = carla.Location(
            self.origin_point.x + (grid_range[0] + grid_range[2]/2) * cell_size[0],
            self.origin_point.y + (grid_range[1] + grid_range[3]/2) * cell_size[1],
            self.origin_point.z
        )
        box_offset = carla.Vector3D(cell_size[0] * grid_range[2]/2,cell_size[1] * grid_range[3]/2, 0.1)
        box_trans = carla.Transform(box_loc, carla.Rotation())

        # Draw the grid cell as a box
        self.world.debug.draw_box(
            box=carla.BoundingBox(box_loc, box_offset),  # Half of the size for each dimension
            rotation=carla.Rotation(pitch=0, yaw=0, roll=0),
            thickness=0.1,  # Box outline thickness
            color=carla.Color(r=255, g=0, b=0),  # Cell color
            life_time=0.15,  # Duration the box is visible
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
            point_from = ((path[n-1][0] + 1 / 2) * self.cell_size[0] + self.origin_point.x,
                          (path[n-1][1] + 1 / 2) * self.cell_size[1] + self.origin_point.y,
                            -90)
            point_to = ((path[n][0] + 1 / 2) * self.cell_size[0] + self.origin_point.x,
                          (path[n][1] + 1 / 2) * self.cell_size[1] + self.origin_point.y,
                            -90)

            self.world.debug.draw_point(
                carla.Location(point_from[0], point_from[1], 0.2),
                size=0.1,
                color=carla.Color(r=30, g=0, b=0),
                life_time=0.12)

            self.world.debug.draw_point(
                carla.Location(point_to[0], point_to[1], 0.2),
                size=0.1,
                color=carla.Color(r=30, g=0, b=0),
                life_time=0.12)

            self.world.debug.draw_line(
                carla.Location(point_from[0], point_from[1], 0.2),
                carla.Location(point_to[0], point_to[1], 0.2),
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
        grid_x = (pos.x - self.origin_point.x) // self.cell_size[0]
        grid_y = (pos.y - self.origin_point.y) // self.cell_size[1]
        return (int(grid_x), int(grid_y))


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






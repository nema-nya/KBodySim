import numpy as np
from ffmpeg_writer import FfmpegWriter
from wgpu_renderer import WgpuRenderer
import dataclasses

number_of_points = 25
point_resolution = 32
window_width = 1024
window_height = 1024
separation = 0.01
point_radius = separation
gravitational_constant = 0.001
substeps = 16
dt = 0.1 / substeps
frame_count = 60
pole_distance = 0.1


@dataclasses.dataclass
class QuadTreeNode:
    bounding_box: tuple
    top_left: any = None
    top_right: any = None
    bottom_left: any = None
    bottom_right: any = None
    points: list = dataclasses.field(default_factory=list)
    mass: float = 0.0
    mass_center: any = dataclasses.field(
        default_factory=lambda: np.zeros(shape=(2,), dtype=float)
    )


class QuadTree:

    def __init__(self, bounding_box, separation, pole_distance):
        self.bounding_box = bounding_box
        self.separation = separation
        self.pole_distance = pole_distance
        self.root = QuadTreeNode(bounding_box)

    def add_point(self, point, position):
        node = self.root
        while True:
            node.mass_center = (node.mass_center * node.mass + position) / (
                node.mass + 1
            )
            node.mass += 1
            bb = node.bounding_box
            if bb[2] - bb[0] < self.separation or bb[3] - bb[1] < self.separation:
                node.points.append((point, position))
                break
            vert = (bb[2] + bb[0]) / 2
            hori = (bb[3] + bb[1]) / 2
            if position[0] < vert and position[1] < hori:
                if node.top_left is None:
                    node.top_left = QuadTreeNode((bb[0], bb[1], vert, hori))
                node = node.top_left
            elif position[0] >= vert and position[1] < hori:
                if node.top_right is None:
                    node.top_right = QuadTreeNode((vert, bb[1], bb[2], hori))
                node = node.top_right
            elif position[0] < vert and position[1] >= hori:
                if node.bottom_left is None:
                    node.bottom_left = QuadTreeNode((bb[0], hori, vert, bb[3]))
                node = node.bottom_left
            elif position[0] >= vert and position[1] >= hori:
                if node.bottom_right is None:
                    node.bottom_right = QuadTreeNode((vert, hori, bb[2], bb[3]))
                node = node.bottom_right

    def get_colisions(self, position):
        queue = [self.root]
        result = []
        while queue:
            node = queue.pop()
            bb = node.bounding_box
            if bb[0] >= position[0] + self.separation:
                continue
            elif bb[1] >= position[1] + self.separation:
                continue
            elif bb[2] < position[0] - self.separation:
                continue
            elif bb[3] < position[1] - self.separation:
                continue
            if node.top_left is not None:
                queue.append(node.top_left)
            if node.top_right is not None:
                queue.append(node.top_right)
            if node.bottom_left is not None:
                queue.append(node.bottom_left)
            if node.bottom_right is not None:
                queue.append(node.bottom_right)

            for point, point_position in node.points:
                if np.linalg.norm(position - point_position) < self.separation:
                    result.append(point)

        return result

    def get_gravity(self, position):
        accel = np.zeros(shape=(2,), dtype=float)
        queue = [self.root]
        while queue:
            node = queue.pop()
            bb = node.bounding_box
            r = node.mass_center - position
            r_len = np.linalg.norm(r)
            if bb[0] >= position[0] + self.pole_distance:
                if r_len > self.separation:
                    accel += r / r_len**3 * gravitational_constant * node.mass
                continue
            elif bb[1] >= position[1] + self.pole_distance:
                if r_len > self.separation:
                    accel += r / r_len**3 * gravitational_constant * node.mass
                continue
            elif bb[2] < position[0] - self.pole_distance:
                if r_len > self.separation:
                    accel += r / r_len**3 * gravitational_constant * node.mass
                continue
            elif bb[3] < position[1] - self.pole_distance:
                if r_len > self.separation:
                    accel += r / r_len**3 * gravitational_constant * node.mass
                continue
            if node.top_left is not None:
                queue.append(node.top_left)
            if node.top_right is not None:
                queue.append(node.top_right)
            if node.bottom_left is not None:
                queue.append(node.bottom_left)
            if node.bottom_right is not None:
                queue.append(node.bottom_right)

            for _, point_position in node.points:
                r = point_position - position
                r_len = np.linalg.norm(r)
                if r_len > self.separation:
                    accel += r / r_len**3 * gravitational_constant
        return accel


class Simulation:
    def __init__(self):
        self.positions = (np.random.rand(number_of_points, 2) * 2 - 1) * (
            1 - separation
        )
        self.positions = self.positions.astype(np.float32)
        for _ in range(substeps):
            quad_tree = QuadTree(
                (*self.positions.min(0), *self.positions.max(0)),
                separation * 2,
                pole_distance,
            )
            for i in range(number_of_points):
                quad_tree.add_point(i, self.positions[i])
            self.resolve_colisions(quad_tree)
        velocity = np.stack([self.positions[:, 1], -self.positions[:, 0]], axis=-1)
        velocity = velocity / np.linalg.norm(velocity, axis=-1, keepdims=True)
        self.prev_positions = self.positions + velocity * dt * 0.06
        self.frame_count = frame_count

    def end_frame(self, image_array):
        ffmpeg_writer.add_frame(image_array)

    def resolve_colisions(self, quad_tree):
        delta = np.zeros_like(self.positions)
        for i in range(number_of_points):
            for j in quad_tree.get_colisions(self.positions[i]):
                if j == i:
                    continue
                r = self.positions[j] - self.positions[i]
                r_len = np.linalg.norm(r)
                if r_len < 2 * separation and r_len > 1e-5:
                    delta[i] -= r / r_len * (2 * separation - r_len) / 2
        self.positions += delta

    def recenter_view(self):
        mean_position = self.positions.mean(0, keepdims=True)
        self.positions -= mean_position
        self.prev_positions -= mean_position

    def start_frame(self):
        for _ in range(substeps):
            quad_tree = QuadTree(
                (*self.positions.min(0), *self.positions.max(0)),
                separation * 2,
                pole_distance,
            )
            for i in range(number_of_points):
                quad_tree.add_point(i, self.positions[i])
            accel = np.zeros(shape=(number_of_points, 2), dtype=np.float32)
            for i in range(number_of_points):
                accel[i] = quad_tree.get_gravity(self.positions[i])
            self.resolve_colisions(quad_tree)
            self.positions, self.prev_positions = (
                2 * self.positions - self.prev_positions + dt**2 * accel,
                self.positions,
            )
        self.recenter_view()
        if self.frame_count > 0:
            self.frame_count -= 1
            return self.positions
        ffmpeg_writer.finish()
        return None


if __name__ == "__main__":
    simulation = Simulation()
    ffmpeg_writer = FfmpegWriter("out.webm", window_width, window_height)
    wgpu_renderer = WgpuRenderer(
        window_width,
        window_height,
        simulation,
        point_resolution,
        point_radius,
        number_of_points,
    )

import numpy as np
from ffmpeg_writer import FfmpegWriter
from wgpu_renderer import WgpuRenderer
from quad_tree_v2 import QuadTreeV2


class Simulation:

    def __init__(self):
        self.number_of_points = 5
        self.point_resolution = 128
        self.window_width = 1024
        self.window_height = 1024
        self.separation = 0.01
        self.point_radius = self.separation
        self.gravitational_constant = 0.001
        self.substeps = 8
        self.dt = 0.001
        self.frame_count = 900
        self.pole_distance = self.separation * 5
        self.spawn_radius = 0.2
        self.initial_spin = 0.003
        self.positions = (
            (np.random.rand(self.number_of_points, 2) * 2 - 1)
            * (1 - self.separation)
            * self.spawn_radius
        )
        self.positions = self.positions.astype(np.float32)
        self.colors = np.ones(shape=(self.number_of_points, 4), dtype=np.float32)
        self.prev_positions = np.zeros_like(self.positions)
        for _ in range(self.substeps):
            quad_tree = QuadTreeV2(
                self.positions,
                self.separation * 2,
                self.pole_distance,
                self.gravitational_constant,
            )
            deltas = quad_tree.get_collisions()
            self.positions += deltas
        velocity = np.stack([self.positions[:, 1], -self.positions[:, 0]], axis=-1)
        velocity = velocity / np.linalg.norm(velocity, axis=-1, keepdims=True)
        self.prev_positions = self.positions + velocity * self.dt * self.initial_spin
        self.frame_count = self.frame_count

    def end_frame(self, image_array):
        ffmpeg_writer.add_frame(image_array)

    def recenter_view(self):
        mean_position = self.positions.mean(0, keepdims=True)
        self.positions -= mean_position
        self.prev_positions -= mean_position

    def start_frame(self):
        # for _ in range(self.substeps):
        #     quad_tree = QuadTreeV2(
        #         self.positions,
        #         self.separation * 2,
        #         self.pole_distance,
        #         self.gravitational_constant,
        #     )
        #     accels = quad_tree.get_gravity()
        #     self.positions, self.prev_positions = (
        #         2 * self.positions - self.prev_positions + self.dt**2 * accels,
        #         self.positions,
        #     )
        #     quad_tree = QuadTreeV2(
        #         self.positions,
        #         self.separation * 2,
        #         self.pole_distance,
        #         self.gravitational_constant,
        #     )
        #     deltas = quad_tree.get_collisions()
        #     self.positions += deltas

        # e_kin = 0
        # for i in range(self.number_of_points):
        #     e_kin += (
        #         np.linalg.norm(self.positions[i] - self.prev_positions[i]) / self.dt
        #     ) ** 2 / 2

        # scale = 100
        # intensity = np.exp(
        #     -scale
        #     * (((self.positions - self.prev_positions) / self.dt) ** 2).sum(-1)
        #     / 2
        # )
        # direction = self.positions - self.prev_positions
        # direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
        # self.colors[:, :2] = (1 - intensity[:, None]) * (
        #     direction / 2 + 1 / 2
        # ) + intensity[:, None] * np.array([1.0, 1.0])[None, :]

        self.recenter_view()
        if self.frame_count > 0:
            self.frame_count -= 1
            return self.positions, self.prev_positions, self.colors
        ffmpeg_writer.finish()
        return None


if __name__ == "__main__":
    simulation = Simulation()
    ffmpeg_writer = FfmpegWriter(
        "out.webm", simulation.window_width, simulation.window_height
    )
    wgpu_renderer = WgpuRenderer(
        simulation.window_width,
        simulation.window_height,
        simulation,
        simulation.point_resolution,
        simulation.point_radius,
        simulation.number_of_points,
    )

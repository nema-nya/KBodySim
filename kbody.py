import numpy as np
from ffmpeg_writer import FfmpegWriter
from wgpu_renderer import WgpuRenderer
from quad_tree_v2 import QuadTreeV2


class Simulation:

    def __init__(self):
        self.number_of_points = 100
        self.point_resolution = 32
        self.window_width = 1024
        self.window_height = 1024
        self.separation = 0.01
        self.point_radius = self.separation
        self.gravitational_constant = 0.001
        self.substeps = 8
        self.dt = 1e-4
        self.frame_count = 900
        self.pole_distance = self.separation * 5
        self.spawn_radius = 0.5
        self.initial_spin = 0.4
        self.init_substeps = 32
        self.ffmpeg_writer = FfmpegWriter(
            "out.webm", self.window_width, self.window_height
        )

        angles = np.random.rand(self.number_of_points) * 2 * np.pi
        radii = np.sqrt(np.random.rand(self.number_of_points)) * self.spawn_radius
        self.positions = (
            np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radii[:, None]
        )
        self.positions = self.positions.astype(np.float32)
        self.colors = np.ones(shape=(self.number_of_points, 4), dtype=np.float32)
        self.prev_positions = np.zeros_like(self.positions)
        velocity = np.stack([self.positions[:, 1], -self.positions[:, 0]], axis=-1)
        self.prev_positions = self.positions + velocity * self.dt * self.initial_spin
        self.frame_count = self.frame_count

    def end_frame(self, image_array):
        self.ffmpeg_writer.add_frame(image_array)

    def start_frame(self):
        if self.frame_count > 0:
            self.frame_count -= 1
            return self.positions, self.prev_positions, self.colors
        self.ffmpeg_writer.finish()
        return None


if __name__ == "__main__":
    simulation = Simulation()
    wgpu_renderer = WgpuRenderer(
        simulation.window_width,
        simulation.window_height,
        simulation,
        simulation.point_resolution,
        simulation.point_radius,
        simulation.number_of_points,
    )

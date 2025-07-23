import numpy as np
from ffmpeg_writer import FfmpegWriter
from wgpu_renderer import WgpuRenderer


class Simulation:

    def __init__(self):
        self.number_of_points = 25
        self.point_resolution = 32
        self.window_width = 1280
        self.window_height = 720
        self.separation = 0.01
        self.point_radius = self.separation
        self.gravitational_constant = 0.001
        self.substeps = 8
        self.dt = 1e-4
        self.frame_count = 600
        self.pole_distance = self.separation * 5
        self.spawn_radius = 0.2
        self.initial_spin = 0.4
        self.init_substeps = 32
        self.max_tree_depth = 4
        self.tree_bb_min = (
            np.array([-1.0, -1.0]) * self.separation * 2 ** (self.max_tree_depth - 1)
        )
        self.tree_bb_max = (
            np.array([1.0, 1.0]) * self.separation * 2 ** (self.max_tree_depth - 1)
        )

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

    def end_frame(self, image_mem_view):
        self.ffmpeg_writer.add_frame(image_mem_view)

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

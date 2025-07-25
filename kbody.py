import numpy as np
from ffmpeg_writer import FfmpegWriter
from wgpu_renderer import WgpuRenderer
import os


class Simulation:

    def __init__(self):
        self.number_of_points = 1000
        self.point_resolution = 32
        self.window_width = 1280
        self.window_height = 720
        self.separation = 0.01
        self.point_radius = self.separation
        self.gravitational_constant = 0.1
        self.substeps = 8
        self.dt = 1e-4
        self.frame_count = 60 * 30
        self.spawn_radius = 0.3
        self.initial_spin = 10
        self.init_substeps = 32
        self.max_tree_depth = 9
        self.theta = 0.5

        self.tree_bb_min = (
            np.array([-1.0, -1.0]) * self.separation * 2 ** (self.max_tree_depth)
        )
        self.tree_bb_max = (
            np.array([1.0, 1.0]) * self.separation * 2 ** (self.max_tree_depth)
        )

        self.ffmpeg_writer = FfmpegWriter(
            os.path.join("outputs", "out.webm"), self.window_width, self.window_height
        )
        self.frame_count = self.frame_count

    def end_frame(self, image_mem_view):
        self.ffmpeg_writer.add_frame(image_mem_view)

    def start_frame(self):
        if self.frame_count > 0:
            self.frame_count -= 1
            return True
        self.ffmpeg_writer.finish()
        return False


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

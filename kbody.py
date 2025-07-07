import numpy as np
from ffmpeg_writer import FfmpegWriter
from wgpu_renderer import WgpuRenderer

number_of_points = 20
point_resolution = 32
vertices = np.empty(
    shape=(point_resolution * number_of_points * 3, 2), dtype=np.float32
)
window_width = 1024
window_height = 1024
separation = 0.01
point_radius = separation
gravitational_constant = 0.001
substeps = 16
dt = 0.1 / substeps


class Simulation:
    def __init__(self):
        self.positions = (np.random.rand(number_of_points, 2) * 2 - 1) * (
            1 - separation
        )
        for _ in range(substeps):
            self.resolve_colisions()
        velocity = np.stack([self.positions[:, 1], -self.positions[:, 0]], axis=-1)
        velocity = velocity / np.linalg.norm(velocity, axis=-1, keepdims=True)
        self.prev_positions = self.positions + velocity * dt * 0.06
        self.frame_count = 300

    def end_frame(self, image_array):
        ffmpeg_writer.add_frame(image_array)

    def resolve_colisions(self):
        delta = np.zeros_like(self.positions)
        for i in range(number_of_points):
            for j in range(number_of_points):
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
            accel = np.zeros(shape=(number_of_points, 2), dtype=float)
            for i in range(number_of_points):
                for j in range(number_of_points):
                    r = self.positions[j] - self.positions[i]
                    r_len = np.linalg.norm(r)
                    if r_len > 2 * separation:
                        accel[i] += r / r_len**3 * gravitational_constant

            self.resolve_colisions()
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
        vertices,
        simulation,
        point_resolution,
        point_radius,
        number_of_points,
    )

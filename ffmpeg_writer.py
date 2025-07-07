import subprocess


class FfmpegWriter:
    def __init__(self, output, window_width, window_height):
        args = [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb32",
            "-s",
            f"{window_width}x{window_height}",
            "-r",
            "60",
            "-i",
            "-",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            "-y",
            output,
        ]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE)

    def add_frame(self, image_array):
        self.proc.stdin.write(image_array.tobytes())

    def finish(self):
        self.proc.stdin.close()
        self.proc.communicate()

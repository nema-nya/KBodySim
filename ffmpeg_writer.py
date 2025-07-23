import subprocess


class FfmpegWriter:
    def __init__(self, output, width, height):
        bytes_per_row = (width * 4 + 255) // 256 * 256 
        args = [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb32",
            "-s",
            f"{bytes_per_row // 4}x{height}",
            "-r",
            "60",
            "-i",
            "-",
            "-vf",
            f"crop={width}:{height}:0:0",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            "-y",
            output,
        ]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE)

    def add_frame(self, image_mem_view):
        self.proc.stdin.write(image_mem_view)

    def finish(self):
        self.proc.stdin.close()
        self.proc.communicate()

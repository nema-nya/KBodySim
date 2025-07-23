import numpy as np
import wgpu


class Buffer:
    def __init__(
        self,
        device,
        shape,
        dtype,
        usage,
        staging=False,
        query=False,
        initial_values=None,
    ):
        self.shape = shape
        self.dtype = dtype
        self.usage = usage
        self.itemsize = dtype.itemsize
        self.nbytes = max(np.prod(shape) * self.itemsize, 32)

        extended_usage = usage

        if staging:
            extended_usage += wgpu.BufferUsage.COPY_DST

        if query:
            extended_usage += wgpu.BufferUsage.COPY_SRC

        self.buffer = device.create_buffer(
            size=self.nbytes,
            usage=extended_usage,
            mapped_at_creation=initial_values is not None,
        )
        if initial_values is not None:
            self.buffer.write_mapped(initial_values)
            self.buffer.unmap()

        if staging:
            self.staging_buffer = device.create_buffer(
                size=self.nbytes,
                usage=wgpu.BufferUsage.MAP_WRITE + wgpu.BufferUsage.COPY_SRC,
            )

        if query:
            self.query_buffer = device.create_buffer(
                size=self.nbytes,
                usage=wgpu.BufferUsage.MAP_READ + wgpu.BufferUsage.COPY_DST,
            )

    def __len__(self):
        return self.shape[0]

    def load_staging(self, command_encoder):
        command_encoder.copy_buffer_to_buffer(
            source=self.staging_buffer,
            source_offset=0,
            destination=self.buffer,
            destination_offset=0,
            size=self.nbytes,
        )

    def store_query(self, command_encoder):
        command_encoder.copy_buffer_to_buffer(
            source=self.buffer,
            source_offset=0,
            destination=self.query_buffer,
            destination_offset=0,
            size=self.nbytes,
        )

    def write_staging(self, values):
        self.staging_buffer.map_sync(wgpu.MapMode.WRITE)
        self.staging_buffer.write_mapped(values)
        self.staging_buffer.unmap()

    def read_query(self):
        self.query_buffer.map_sync(wgpu.MapMode.READ)
        values = self.query_buffer.read_mapped()
        self.query_buffer.unmap()
        return values

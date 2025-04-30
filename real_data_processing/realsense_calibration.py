import pyrealsense2 as rs

# Start pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Wait for a few frames to stabilize the stream
for _ in range(5):
    frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# Get intrinsics
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# Print intrinsics
print("=== RGB Camera Intrinsics ===")
print(f"Resolution: {intrinsics.width}x{intrinsics.height}")
print(f"Focal Lengths: fx={intrinsics.fx}, fy={intrinsics.fy}")
print(f"Principal Point: ppx={intrinsics.ppx}, ppy={intrinsics.ppy}")
print(f"Distortion Model: {intrinsics.model}")
print(f"Distortion Coefficients: {intrinsics.coeffs}")

# Clean up
pipeline.stop()

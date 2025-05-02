import unittest
import sys
import os
import torch
import tempfile
import imageio

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_util import save_video

class TestSaveVideo(unittest.TestCase):
    def test_save_video_creates_file(self):
        """Test that save_video creates a file with the expected format."""
        # Create a simple tensor with shape [1, 3, 16, 64, 64]
        # This represents a batch of 1 video with 3 channels (RGB), 16 frames, and 64x64 resolution
        frames = torch.zeros(1, 3, 16, 64, 64)
        
        # Call save_video with a test seed
        test_seed = "123456"
        output_path = save_video(frames, seed=test_seed)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check that the filename contains the seed
        self.assertIn(test_seed, os.path.basename(output_path))
        
        # Check that the file has the .mp4 extension
        self.assertTrue(output_path.endswith(".mp4"))
        
        # Try to open the video file to verify it's a valid video
        try:
            reader = imageio.get_reader(output_path)
            # Get the number of frames
            num_frames = len(reader)
            # Check that the number of frames matches what we expect
            self.assertEqual(num_frames, 16)
            reader.close()
        except Exception as e:
            self.fail(f"Failed to open video file: {e}")
        
        # Clean up the file
        os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
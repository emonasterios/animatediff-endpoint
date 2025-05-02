import unittest
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_util import check_data_format

class TestCheckDataFormat(unittest.TestCase):
    def test_minimal_input(self):
        """Test with only the required prompt parameter."""
        input_data = {"prompt": "test prompt"}
        result = check_data_format(input_data)
        self.assertEqual(result["prompt"], "test prompt")
        self.assertIsNone(result["steps"])
        self.assertIsNone(result["width"])
        self.assertIsNone(result["height"])
        self.assertIsNone(result["n_prompt"])
        self.assertIsNone(result["guidance_scale"])
        self.assertIsNone(result["seed"])
        self.assertIsNone(result["base_model"])
        self.assertIsNone(result["base_loras"])
        self.assertIsNone(result["motion_lora"])

    def test_complete_input(self):
        """Test with all parameters provided."""
        input_data = {
            "prompt": "test prompt",
            "steps": 25,
            "width": 512,
            "height": 512,
            "n_prompt": "negative prompt",
            "guidance_scale": 7.5,
            "seed": 42,
            "base_model": "test_model",
            "base_loras": {"lora1": ["test_lora.safetensors", 0.5]},
            "motion_lora": ["test_motion_lora.safetensors", 0.8]
        }
        result = check_data_format(input_data)
        self.assertEqual(result["prompt"], "test prompt")
        self.assertEqual(result["steps"], 25)
        self.assertEqual(result["width"], 512)
        self.assertEqual(result["height"], 512)
        self.assertEqual(result["n_prompt"], "negative prompt")
        self.assertEqual(result["guidance_scale"], 7.5)
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["base_model"], "test_model")
        self.assertEqual(result["base_loras"], {"lora1": ["test_lora.safetensors", 0.5]})
        self.assertEqual(result["motion_lora"], ["test_motion_lora.safetensors", 0.8])

    def test_missing_prompt(self):
        """Test that an error is raised when prompt is missing."""
        input_data = {"steps": 25}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("The input must contain a prompt" in str(context.exception))

    def test_invalid_prompt_type(self):
        """Test that an error is raised when prompt is not a string."""
        input_data = {"prompt": 123}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("prompt must be a string" in str(context.exception))

    def test_invalid_steps_type(self):
        """Test that an error is raised when steps is not an integer."""
        input_data = {"prompt": "test prompt", "steps": "25"}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("steps must be an integer" in str(context.exception))

    def test_invalid_width_type(self):
        """Test that an error is raised when width is not an integer."""
        input_data = {"prompt": "test prompt", "width": "512"}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("width must be an integer" in str(context.exception))

    def test_invalid_height_type(self):
        """Test that an error is raised when height is not an integer."""
        input_data = {"prompt": "test prompt", "height": "512"}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("height must be an integer" in str(context.exception))

    def test_invalid_n_prompt_type(self):
        """Test that an error is raised when n_prompt is not a string."""
        input_data = {"prompt": "test prompt", "n_prompt": 123}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("n_prompt must be a string" in str(context.exception))

    def test_invalid_guidance_scale_type(self):
        """Test that an error is raised when guidance_scale is not a number."""
        input_data = {"prompt": "test prompt", "guidance_scale": "7.5"}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("guidance_scale must be a float or an integer" in str(context.exception))

    def test_invalid_seed_type(self):
        """Test that an error is raised when seed is not an integer."""
        input_data = {"prompt": "test prompt", "seed": "42"}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("seed must be an integer" in str(context.exception))

    def test_invalid_base_model_type(self):
        """Test that an error is raised when base_model is not a string."""
        input_data = {"prompt": "test prompt", "base_model": 123}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("base_model must be a string" in str(context.exception))

    def test_invalid_base_loras_type(self):
        """Test that an error is raised when base_loras is not a dictionary."""
        input_data = {"prompt": "test prompt", "base_loras": "not a dict"}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("base_loras must be a dictionary" in str(context.exception))

    def test_invalid_base_loras_key_type(self):
        """Test that an error is raised when base_loras keys are not strings."""
        input_data = {"prompt": "test prompt", "base_loras": {123: ["test_lora.safetensors", 0.5]}}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("base_loras keys must be strings" in str(context.exception))

    def test_invalid_base_loras_value_type(self):
        """Test that an error is raised when base_loras values are not lists."""
        input_data = {"prompt": "test prompt", "base_loras": {"lora1": "not a list"}}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("base_loras values must be lists" in str(context.exception))

    def test_invalid_base_loras_value_length(self):
        """Test that an error is raised when base_loras values are not lists of length 2."""
        input_data = {"prompt": "test prompt", "base_loras": {"lora1": ["test_lora.safetensors"]}}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("base_loras values must be lists of length 2" in str(context.exception))

    def test_invalid_base_loras_value_first_element_type(self):
        """Test that an error is raised when the first element of base_loras values is not a string."""
        input_data = {"prompt": "test prompt", "base_loras": {"lora1": [123, 0.5]}}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("base_loras values must be lists of strings" in str(context.exception))

    def test_invalid_base_loras_value_second_element_type(self):
        """Test that an error is raised when the second element of base_loras values is not a float."""
        input_data = {"prompt": "test prompt", "base_loras": {"lora1": ["test_lora.safetensors", "0.5"]}}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("base_loras values must be lists of floats" in str(context.exception))

    def test_invalid_motion_lora_type(self):
        """Test that an error is raised when motion_lora is not a list."""
        input_data = {"prompt": "test prompt", "motion_lora": "not a list"}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("motion_lora must be a list" in str(context.exception))

    def test_invalid_motion_lora_length(self):
        """Test that an error is raised when motion_lora is not a list of length 2."""
        input_data = {"prompt": "test prompt", "motion_lora": ["test_motion_lora.safetensors"]}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("motion_lora must be a list of length 2" in str(context.exception))

    def test_invalid_motion_lora_first_element_type(self):
        """Test that an error is raised when the first element of motion_lora is not a string."""
        input_data = {"prompt": "test prompt", "motion_lora": [123, 0.8]}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("motion_lora must be a list of strings" in str(context.exception))

    def test_invalid_motion_lora_second_element_type(self):
        """Test that an error is raised when the second element of motion_lora is not a number."""
        input_data = {"prompt": "test prompt", "motion_lora": ["test_motion_lora.safetensors", "0.8"]}
        with self.assertRaises(ValueError) as context:
            check_data_format(input_data)
        self.assertTrue("motion_lora must be a list of floats" in str(context.exception))

    def test_guidance_scale_as_int(self):
        """Test that guidance_scale can be an integer."""
        input_data = {"prompt": "test prompt", "guidance_scale": 7}
        result = check_data_format(input_data)
        self.assertEqual(result["guidance_scale"], 7)

    def test_motion_lora_second_element_as_int(self):
        """Test that the second element of motion_lora can be an integer."""
        input_data = {"prompt": "test prompt", "motion_lora": ["test_motion_lora.safetensors", 1]}
        result = check_data_format(input_data)
        self.assertEqual(result["motion_lora"], ["test_motion_lora.safetensors", 1])


if __name__ == "__main__":
    unittest.main()
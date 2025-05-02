import unittest
import sys
import os
import torch
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_util import AnimateDiff

class TestAnimateDiff(unittest.TestCase):
    @patch('inference_util.init_pipeline')
    @patch('inference_util.OmegaConf.load')
    def test_get_model_params_with_person_keyword(self, mock_load, mock_init_pipeline):
        """Test that _get_model_params correctly identifies person-related prompts."""
        # Mock the OmegaConf.load to return a config with Person and Default sections
        mock_config = MagicMock()
        mock_config.Person = MagicMock()
        mock_config.Person.base_model = "person_model"
        mock_config.Person.base_loras = {"person_lora": ["person_lora.safetensors", 0.5]}
        mock_config.Person.motion_lora = ["person_motion_lora.safetensors", 0.8]
        mock_config.Person.prompt = "person style"
        mock_config.Person.n_prompt = "person negative prompt"
        mock_config.Person.width = 512
        mock_config.Person.height = 512
        
        mock_config.Default = MagicMock()
        mock_config.Default.base_model = "default_model"
        mock_config.Default.base_loras = {"default_lora": ["default_lora.safetensors", 0.5]}
        mock_config.Default.motion_lora = ["default_motion_lora.safetensors", 0.8]
        mock_config.Default.prompt = "default style"
        mock_config.Default.n_prompt = "default negative prompt"
        mock_config.Default.width = 768
        mock_config.Default.height = 512
        
        mock_load.return_value = mock_config
        
        # Create an AnimateDiff instance with the mocked config
        animatediff = AnimateDiff(version="v2")
        
        # Test with a prompt containing a person-related keyword
        prompt, width, height, n_prompt, base_model, base_loras, motion_lora = animatediff._get_model_params(
            prompt="a beautiful girl in a garden",
            width=None,
            height=None,
            n_prompt=None,
            base_model=None,
            base_loras=None,
            motion_lora=None
        )
        
        # Check that the person model was selected
        self.assertEqual(base_model, "person_model")
        self.assertEqual(base_loras, {"person_lora": ["person_lora.safetensors", 0.5]})
        self.assertEqual(motion_lora, ["person_motion_lora.safetensors", 0.8])
        self.assertEqual(n_prompt, "person negative prompt")
        self.assertEqual(width, 512)
        self.assertEqual(height, 512)
        self.assertEqual(prompt, "a beautiful girl in a garden, person style")
    
    @patch('inference_util.init_pipeline')
    @patch('inference_util.OmegaConf.load')
    def test_get_model_params_without_person_keyword(self, mock_load, mock_init_pipeline):
        """Test that _get_model_params correctly handles prompts without person-related keywords."""
        # Mock the OmegaConf.load to return a config with Person and Default sections
        mock_config = MagicMock()
        mock_config.Person = MagicMock()
        mock_config.Person.base_model = "person_model"
        mock_config.Person.base_loras = {"person_lora": ["person_lora.safetensors", 0.5]}
        mock_config.Person.motion_lora = ["person_motion_lora.safetensors", 0.8]
        mock_config.Person.prompt = "person style"
        mock_config.Person.n_prompt = "person negative prompt"
        mock_config.Person.width = 512
        mock_config.Person.height = 512
        
        mock_config.Default = MagicMock()
        mock_config.Default.base_model = "default_model"
        mock_config.Default.base_loras = {"default_lora": ["default_lora.safetensors", 0.5]}
        mock_config.Default.motion_lora = ["default_motion_lora.safetensors", 0.8]
        mock_config.Default.prompt = "default style"
        mock_config.Default.n_prompt = "default negative prompt"
        mock_config.Default.width = 768
        mock_config.Default.height = 512
        
        mock_load.return_value = mock_config
        
        # Create an AnimateDiff instance with the mocked config
        animatediff = AnimateDiff(version="v2")
        
        # Test with a prompt that doesn't contain a person-related keyword
        prompt, width, height, n_prompt, base_model, base_loras, motion_lora = animatediff._get_model_params(
            prompt="a beautiful landscape with mountains",
            width=None,
            height=None,
            n_prompt=None,
            base_model=None,
            base_loras=None,
            motion_lora=None
        )
        
        # Check that the default model was selected
        self.assertEqual(base_model, "default_model")
        self.assertEqual(base_loras, {"default_lora": ["default_lora.safetensors", 0.5]})
        self.assertEqual(motion_lora, ["default_motion_lora.safetensors", 0.8])
        self.assertEqual(n_prompt, "default negative prompt")
        self.assertEqual(width, 768)
        self.assertEqual(height, 512)
        self.assertEqual(prompt, "a beautiful landscape with mountains, default style")
    
    @patch('inference_util.init_pipeline')
    @patch('inference_util.OmegaConf.load')
    def test_get_model_params_with_custom_params(self, mock_load, mock_init_pipeline):
        """Test that _get_model_params correctly uses custom parameters when provided."""
        # Mock the OmegaConf.load to return a config with Person and Default sections
        mock_config = MagicMock()
        mock_config.Person = MagicMock()
        mock_config.Person.base_model = "person_model"
        mock_config.Person.base_loras = {"person_lora": ["person_lora.safetensors", 0.5]}
        mock_config.Person.motion_lora = ["person_motion_lora.safetensors", 0.8]
        mock_config.Person.prompt = "person style"
        mock_config.Person.n_prompt = "person negative prompt"
        mock_config.Person.width = 512
        mock_config.Person.height = 512
        
        mock_config.Default = MagicMock()
        mock_config.Default.base_model = "default_model"
        mock_config.Default.base_loras = {"default_lora": ["default_lora.safetensors", 0.5]}
        mock_config.Default.motion_lora = ["default_motion_lora.safetensors", 0.8]
        mock_config.Default.prompt = "default style"
        mock_config.Default.n_prompt = "default negative prompt"
        mock_config.Default.width = 768
        mock_config.Default.height = 512
        
        mock_load.return_value = mock_config
        
        # Create an AnimateDiff instance with the mocked config
        animatediff = AnimateDiff(version="v2")
        
        # Test with custom parameters
        custom_base_model = "custom_model"
        custom_base_loras = {"custom_lora": ["custom_lora.safetensors", 0.7]}
        custom_motion_lora = ["custom_motion_lora.safetensors", 0.9]
        custom_width = 1024
        custom_height = 768
        custom_n_prompt = "custom negative prompt"
        
        prompt, width, height, n_prompt, base_model, base_loras, motion_lora = animatediff._get_model_params(
            prompt="a beautiful landscape with mountains",
            width=custom_width,
            height=custom_height,
            n_prompt=custom_n_prompt,
            base_model=custom_base_model,
            base_loras=custom_base_loras,
            motion_lora=custom_motion_lora
        )
        
        # Check that the custom parameters were used
        self.assertEqual(base_model, custom_base_model)
        self.assertEqual(base_loras, custom_base_loras)
        self.assertEqual(motion_lora, custom_motion_lora)
        self.assertEqual(n_prompt, custom_n_prompt)
        self.assertEqual(width, custom_width)
        self.assertEqual(height, custom_height)
        self.assertEqual(prompt, "a beautiful landscape with mountains")
    
    @patch('inference_util.init_pipeline')
    @patch('inference_util.OmegaConf.load')
    def test_get_model_params_with_prompt_ending_with_period(self, mock_load, mock_init_pipeline):
        """Test that _get_model_params correctly handles prompts ending with a period."""
        # Mock the OmegaConf.load to return a config with Person and Default sections
        mock_config = MagicMock()
        mock_config.Default = MagicMock()
        mock_config.Default.base_model = "default_model"
        mock_config.Default.base_loras = {"default_lora": ["default_lora.safetensors", 0.5]}
        mock_config.Default.motion_lora = ["default_motion_lora.safetensors", 0.8]
        mock_config.Default.prompt = "default style"
        mock_config.Default.n_prompt = "default negative prompt"
        mock_config.Default.width = 768
        mock_config.Default.height = 512
        
        mock_load.return_value = mock_config
        
        # Create an AnimateDiff instance with the mocked config
        animatediff = AnimateDiff(version="v2")
        
        # Test with a prompt ending with a period
        prompt, width, height, n_prompt, base_model, base_loras, motion_lora = animatediff._get_model_params(
            prompt="a beautiful landscape with mountains.",
            width=None,
            height=None,
            n_prompt=None,
            base_model=None,
            base_loras=None,
            motion_lora=None
        )
        
        # Check that the period was removed and the default style was appended
        self.assertEqual(prompt, "a beautiful landscape with mountains, default style")


if __name__ == "__main__":
    unittest.main()
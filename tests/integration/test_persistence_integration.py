"""Integration tests for persistence functionality with real providers."""

import os
import tempfile
from pathlib import Path

import pytest

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


class TestPersistenceIntegration:
    """Integration tests for module persistence with real providers."""

    def setup_method(self):
        """Set up test environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not available")

        # Set up provider
        self.provider = create_provider("openai", model="gpt-4.1")
        register_provider(self.provider, set_default=True)

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_save_and_load_basic_module(self):
        """Test saving and loading a basic module with real predictions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_module.json"

            # Create and configure module
            original_module = Predict("question: str -> answer: str")
            original_module.config["temperature"] = 0.5
            original_module.metadata["test_id"] = "integration_test"

            # Make a prediction to validate it works
            result = await original_module(question="What is 2+2?")
            assert result.outputs["answer"]

            # Save the module
            original_module.save(str(save_path))

            # Load the module
            loaded_module = Predict.load(str(save_path))

            # Verify configuration is preserved
            assert loaded_module.config["temperature"] == 0.5
            assert loaded_module.metadata["test_id"] == "integration_test"

            # Test that loaded module works
            loaded_result = await loaded_module(question="What is 3+3?")
            assert loaded_result.outputs["answer"]

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_save_and_load_with_provider_config(self):
        """Test persistence includes provider configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "module_with_provider.json"

            # Create module with specific provider
            module = Predict("task: str -> result: str", provider=self.provider)

            # Make a prediction
            result = await module(task="Summarize: The weather is nice today.")
            assert result.outputs["result"]

            # Save with provider configuration
            module.save(str(save_path), include_provider=True)

            # Verify save file contains provider info
            import json

            with open(save_path) as f:
                data = json.load(f)

            assert "provider_config" in data
            provider_config = data["provider_config"]
            assert provider_config["name"] == "openai"
            assert provider_config["model"] == "gpt-4.1"

            # Load module (without automatic provider setup)
            loaded_module = Predict.load(str(save_path), setup_provider=False)

            # Should still work with current provider
            loaded_result = await loaded_module(task="Count: one, two, three")
            assert loaded_result.outputs["result"]

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_save_load_complex_signature(self):
        """Test persistence with complex signature types."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "complex_signature.json"

            # Create module with complex signature
            signature_str = "context: str, question: str -> answer: str, confidence: float"
            module = Predict(signature_str)

            # Test original module
            result = await module(
                context="Python is a programming language.", question="What is Python?"
            )
            assert result.outputs["answer"]
            # Note: confidence might not be provided by LLM, that's ok for this test

            # Save and load
            module.save(str(save_path))
            loaded_module = Predict.load(str(save_path))

            # Test loaded module
            loaded_result = await loaded_module(
                context="Dogs are mammals.", question="What are dogs?"
            )
            assert loaded_result.outputs["answer"]

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_persistence_version_compatibility(self):
        """Test version compatibility warnings and handling."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "version_test.json"

            # Create and save module
            module = Predict("text: str -> summary: str")
            module.save(str(save_path))

            # Modify saved file to have different version
            import json

            with open(save_path) as f:
                data = json.load(f)

            data["logillm_version"] = "0.0.1"  # Fake older version

            with open(save_path, "w") as f:
                json.dump(data, f)

            # Load with version warning
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loaded_module = Predict.load(str(save_path))

                # Should get a version warning
                assert len(w) >= 1
                version_warnings = [
                    warning for warning in w if "version" in str(warning.message).lower()
                ]
                assert len(version_warnings) > 0

            # Module should still work despite version difference
            result = await loaded_module(text="This is a test message for summarization.")
            assert result.outputs["summary"]

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_round_trip_preservation(self):
        """Test that save/load preserves module behavior exactly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "round_trip.json"

            # Create module with specific configuration
            original_module = Predict("prompt: str -> response: str")
            original_module.config.update(
                {
                    "temperature": 0.1,  # Low temperature for consistency
                    "max_tokens": 50,
                }
            )

            # Make prediction with original
            test_input = "Say exactly: 'Hello, World!'"
            original_result = await original_module(prompt=test_input)

            # Save and load
            original_module.save(str(save_path))
            loaded_module = Predict.load(str(save_path))

            # Test loaded module behavior
            loaded_result = await loaded_module(prompt=test_input)

            # Both should have produced responses
            assert original_result.outputs["response"]
            assert loaded_result.outputs["response"]

            # Configuration should be identical
            assert loaded_module.config["temperature"] == 0.1
            assert loaded_module.config["max_tokens"] == 50

    @pytest.mark.integration
    @pytest.mark.openai
    def test_file_system_integration(self):
        """Test persistence works with real file system operations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test nested directory creation
            nested_path = Path(tmp_dir) / "deep" / "nested" / "dirs" / "module.json"

            module = Predict("input -> output")
            module.metadata["created_in"] = "nested_directory"

            # Should create all intermediate directories
            module.save(str(nested_path))
            assert nested_path.exists()

            # Test loading from nested path
            loaded_module = Predict.load(str(nested_path))
            assert loaded_module.metadata["created_in"] == "nested_directory"

            # Test overwriting existing file
            module.metadata["version"] = "2.0"
            module.save(str(nested_path))  # Overwrite

            reloaded_module = Predict.load(str(nested_path))
            assert reloaded_module.metadata["version"] == "2.0"

    @pytest.mark.integration
    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_persistence_error_handling(self):
        """Test error handling in persistence operations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test loading non-existent file
            from logillm.core.persistence import PersistenceError

            with pytest.raises(PersistenceError, match="File not found"):
                Predict.load(str(Path(tmp_dir) / "nonexistent.json"))

            # Test loading corrupted file
            corrupted_path = Path(tmp_dir) / "corrupted.json"
            with open(corrupted_path, "w") as f:
                f.write("invalid json content {")

            with pytest.raises(Exception):  # JSON decode error
                Predict.load(str(corrupted_path))

            # Test saving to read-only location (if possible)
            module = Predict("test -> test")

            # Create a file and make directory read-only (Unix only)
            if hasattr(os, "chmod"):
                readonly_dir = Path(tmp_dir) / "readonly"
                readonly_dir.mkdir()
                os.chmod(readonly_dir, 0o444)  # Read-only

                try:
                    with pytest.raises(PermissionError):
                        module.save(str(readonly_dir / "test.json"))
                finally:
                    # Restore permissions for cleanup
                    os.chmod(readonly_dir, 0o755)

"""Unit tests for persistence functionality."""

import json
import warnings
from unittest.mock import Mock, patch

import pytest

from logillm.core.demos import Demo, DemoManager
from logillm.core.modules import ModuleState
from logillm.core.persistence import ModuleLoader, ModuleSaver, PersistenceError
from logillm.core.predict import Predict
from logillm.core.types import SerializationFormat
from logillm.exceptions import LogiLLMError


class TestModuleSaver:
    """Test ModuleSaver functionality."""

    def test_save_module_basic(self, tmp_path):
        """Test basic module saving."""
        # Create a simple module
        module = Predict("input -> output")
        module.config["temperature"] = 0.8
        module.metadata["test"] = "value"

        save_path = tmp_path / "test_module.json"

        # Save the module
        ModuleSaver.save_module(module, str(save_path))

        # Check file was created
        assert save_path.exists()

        # Load and verify data
        with open(save_path) as f:
            data = json.load(f)

        assert data["logillm_version"] == "0.1.0"
        assert data["module_type"] == "Predict"
        assert "save_timestamp" in data
        assert data["config"]["temperature"] == 0.8
        assert data["metadata"]["test"] == "value"
        assert data["state"] == ModuleState.INITIALIZED.value

    def test_save_module_with_signature(self, tmp_path):
        """Test saving module with signature."""
        module = Predict("question: str -> answer: str")
        save_path = tmp_path / "module_with_sig.json"

        ModuleSaver.save_module(module, str(save_path))

        with open(save_path) as f:
            data = json.load(f)

        assert data["signature"] is not None
        assert "input_fields" in data["signature"]
        assert "output_fields" in data["signature"]

    def test_save_module_with_demo_manager(self, tmp_path):
        """Test saving module with demo manager."""
        module = Predict("input -> output")

        # Add demo manager with demos
        demo_manager = DemoManager(max_demos=3)
        demo = Demo(inputs={"input": "test question"}, outputs={"output": "test answer"}, score=0.9)
        demo_manager.demos.append(demo)
        module.demo_manager = demo_manager

        save_path = tmp_path / "module_with_demos.json"
        ModuleSaver.save_module(module, str(save_path))

        with open(save_path) as f:
            data = json.load(f)

        assert "demo_manager" in data
        assert len(data["demo_manager"]["demos"]) == 1
        assert data["demo_manager"]["max_demos"] == 3

    def test_save_module_with_provider(self, tmp_path):
        """Test saving module with provider configuration."""
        # Mock provider
        provider = Mock()
        provider.name = "openai"
        provider.model = "gpt-4.1"
        provider.config = {"temperature": 0.7, "api_key": "secret"}
        provider.__class__.__module__ = "logillm.providers.openai"
        provider.__class__.__qualname__ = "OpenAIProvider"

        module = Predict("input -> output")
        module.provider = provider

        save_path = tmp_path / "module_with_provider.json"
        ModuleSaver.save_module(module, str(save_path), include_provider=True)

        with open(save_path) as f:
            data = json.load(f)

        assert "provider_config" in data
        provider_config = data["provider_config"]
        assert provider_config["name"] == "openai"
        assert provider_config["model"] == "gpt-4.1"
        # API key should be filtered out
        assert "api_key" not in provider_config["config"]
        assert "temperature" in provider_config["config"]

    def test_save_module_creates_directories(self, tmp_path):
        """Test that saving creates intermediate directories."""
        module = Predict("input -> output")
        save_path = tmp_path / "nested" / "dir" / "module.json"

        ModuleSaver.save_module(module, str(save_path))

        assert save_path.exists()

    def test_save_module_unsupported_format(self):
        """Test error on unsupported serialization format."""
        module = Predict("input -> output")

        with pytest.raises(PersistenceError, match="Format .* not yet supported"):
            ModuleSaver.save_module(module, "test.yaml", format=SerializationFormat.YAML)


class TestModuleLoader:
    """Test ModuleLoader functionality."""

    def test_load_module_basic(self, tmp_path):
        """Test basic module loading."""
        # Create test data
        save_data = {
            "logillm_version": "0.1.0",
            "save_timestamp": "2024-01-01T12:00:00",
            "module_type": "Predict",
            "signature": None,
            "config": {"temperature": 0.8},
            "metadata": {"test": "value"},
            "state": "initialized",
        }

        save_path = tmp_path / "test_module.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f)

        # Load the module
        module = ModuleLoader.load_module(str(save_path))

        assert isinstance(module, Predict)
        assert module.config["temperature"] == 0.8
        assert module.metadata["test"] == "value"

    def test_load_module_with_signature(self, tmp_path):
        """Test loading module with signature."""
        # Create test signature data
        sig_data = {
            "type": "BaseSignature",
            "instructions": None,
            "metadata": {},
            "input_fields": {
                "question": {
                    "field_type": "input",
                    "python_type": "str",
                    "description": "",
                    "required": True,
                    "default": None,
                    "constraints": {},
                    "metadata": {},
                }
            },
            "output_fields": {
                "answer": {
                    "field_type": "output",
                    "python_type": "str",
                    "description": "",
                    "required": True,
                    "default": None,
                    "constraints": {},
                    "metadata": {},
                }
            },
        }

        save_data = {
            "logillm_version": "0.1.0",
            "save_timestamp": "2024-01-01T12:00:00",
            "module_type": "Predict",
            "signature": sig_data,
            "config": {},
            "metadata": {},
            "state": "initialized",
        }

        save_path = tmp_path / "test_module.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f)

        # Load the module
        with patch("logillm.core.signatures.utils.make_field_spec_from_dict") as mock_make_field:
            # Mock field spec creation
            mock_spec = Mock()
            mock_make_field.return_value = mock_spec

            module = ModuleLoader.load_module(str(save_path))
            assert isinstance(module, Predict)

    def test_load_module_with_demo_manager(self, tmp_path):
        """Test loading module with demo manager."""
        demo_data = {
            "demos": [
                {
                    "inputs": {"input": "test"},
                    "outputs": {"output": "result"},
                    "score": 0.9,
                    "metadata": {},
                }
            ],
            "teacher_demos": [],
            "max_demos": 3,
            "selection_strategy": "best",
        }

        save_data = {
            "logillm_version": "0.1.0",
            "save_timestamp": "2024-01-01T12:00:00",
            "module_type": "Predict",
            "signature": None,
            "config": {},
            "metadata": {},
            "state": "initialized",
            "demo_manager": demo_data,
        }

        save_path = tmp_path / "test_module.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f)

        module = ModuleLoader.load_module(str(save_path))

        assert hasattr(module, "demo_manager")
        assert len(module.demo_manager.demos) == 1
        assert module.demo_manager.max_demos == 3

    def test_load_module_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(PersistenceError, match="File not found"):
            ModuleLoader.load_module("nonexistent.json")

    def test_load_module_version_warning(self, tmp_path):
        """Test version compatibility warning."""
        save_data = {
            "logillm_version": "0.0.1",  # Different version
            "save_timestamp": "2024-01-01T12:00:00",
            "module_type": "Predict",
            "signature": None,
            "config": {},
            "metadata": {},
            "state": "initialized",
        }

        save_path = tmp_path / "test_module.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ModuleLoader.load_module(str(save_path))

            assert len(w) == 1
            assert "version" in str(w[0].message).lower()

    def test_load_module_strict_version_error(self, tmp_path):
        """Test strict version enforcement."""
        save_data = {
            "logillm_version": "0.0.1",  # Different version
            "save_timestamp": "2024-01-01T12:00:00",
            "module_type": "Predict",
            "signature": None,
            "config": {},
            "metadata": {},
            "state": "initialized",
        }

        save_path = tmp_path / "test_module.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f)

        with pytest.raises(PersistenceError, match="Version mismatch"):
            ModuleLoader.load_module(str(save_path), strict_version=True)

    def test_load_module_unsupported_type(self, tmp_path):
        """Test error for unsupported module type."""
        save_data = {
            "logillm_version": "0.1.0",
            "save_timestamp": "2024-01-01T12:00:00",
            "module_type": "UnsupportedModule",
            "signature": None,
            "config": {},
            "metadata": {},
            "state": "initialized",
        }

        save_path = tmp_path / "test_module.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f)

        with pytest.raises(PersistenceError, match="Unsupported module type"):
            ModuleLoader.load_module(str(save_path))


class TestMonkeyPatchedMethods:
    """Test the monkey-patched save/load methods on Module."""

    def test_module_save_method(self, tmp_path):
        """Test the save method added to Module."""
        module = Predict("input -> output")
        save_path = tmp_path / "module.json"

        # Test the monkey-patched save method
        module.save(str(save_path))

        assert save_path.exists()

        # Verify content
        with open(save_path) as f:
            data = json.load(f)
        assert data["module_type"] == "Predict"

    def test_module_load_classmethod(self, tmp_path):
        """Test the load classmethod added to Module."""
        # Create a saved module
        module = Predict("input -> output")
        module.config["test"] = "value"
        save_path = tmp_path / "module.json"
        module.save(str(save_path))

        # Test loading via the monkey-patched class method
        loaded_module = Predict.load(str(save_path))

        assert isinstance(loaded_module, Predict)
        assert loaded_module.config["test"] == "value"


class TestPersistenceError:
    """Test PersistenceError exception."""

    def test_persistence_error_inheritance(self):
        """Test that PersistenceError inherits from LogiLLMError."""
        error = PersistenceError("test error")
        assert isinstance(error, LogiLLMError)
        assert str(error) == "test error"

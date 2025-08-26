"""Integration tests for the assertion system demonstrating real usage scenarios."""

import json

import pytest

from logillm.core.assertions import (
    Assert,
    AssertionError,
    AssertionSeverity,
    BacktrackHandler,
    BacktrackStrategy,
    SuggestionGenerator,
    assert_module_output,
    assertion_context,
)
from logillm.core.modules import BaseModule
from logillm.core.types import Prediction, Usage


class TestJSONValidationWorkflow:
    """Test JSON validation workflow with assertions."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_json_validation_success(self):
        """Test successful JSON validation workflow."""
        async with assertion_context("json_validation") as context:
            # Add JSON format assertion
            json_assertion = Assert.format(
                "valid_json",
                format_type="json",
                required_keys=["name", "age", "city"],
                severity=AssertionSeverity.HARD,
            )
            context.add_assertion(json_assertion)

            # Test valid JSON
            valid_data = {"name": "John Doe", "age": 30, "city": "New York"}
            results = context.check_all(valid_data)

            assert len(results) == 1
            assert results[0].passed is True
            assert not context.has_failures()

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_json_validation_failure_and_recovery(self):
        """Test JSON validation failure and manual recovery."""
        async with assertion_context("json_validation") as context:
            json_assertion = Assert.format(
                "valid_json",
                format_type="json",
                required_keys=["name", "age"],
                severity=AssertionSeverity.HARD,
            )
            context.add_assertion(json_assertion)

            # Test invalid JSON (missing required key)
            invalid_data = {"name": "John Doe"}
            results = context.check_all(invalid_data)

            assert len(results) == 1
            assert results[0].passed is False
            assert context.has_failures()

            failures = context.get_failures()
            assert "Missing required JSON keys: ['age']" in failures[0].message
            assert "Add missing keys: ['age']" in failures[0].suggestions

            # Manually fix the data based on suggestions
            fixed_data = {"name": "John Doe", "age": 25}
            context.clear_results()

            results = context.check_all(fixed_data)
            assert results[0].passed is True


class TestDataValidationPipeline:
    """Test data validation pipeline with multiple assertion types."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_multi_assertion_pipeline(self):
        """Test pipeline with multiple assertion types."""
        async with assertion_context("data_pipeline") as context:
            # Add multiple assertions
            assertions = [
                Assert.constraint("non_empty", non_empty=True),
                Assert.value("string_type", expected_type=str),
                Assert.constraint("length_check", min_length=5, max_length=100),
                Assert.format(
                    "email_format",
                    format_type="regex",
                    pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                ),
            ]

            for assertion in assertions:
                context.add_assertion(assertion)

            # Test valid email
            valid_email = "user@example.com"
            results = context.check_all(valid_email)

            assert len(results) == 4
            assert all(r.passed for r in results)

            # Test invalid email
            context.clear_results()
            invalid_email = "invalid"
            results = context.check_all(invalid_email)

            # Should pass some checks but fail regex
            passed_results = [r for r in results if r.passed]
            failed_results = [r for r in results if not r.passed]

            assert len(passed_results) == 3  # non_empty, string_type, length_check
            assert len(failed_results) == 1  # email_format
            assert "does not match regex pattern" in failed_results[0].message


class TestModuleIntegration:
    """Test assertion integration with LogiLLM modules."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_module_with_assertions(self, isolated_module):
        """Test module with built-in assertions."""

        # Create a simple module
        class TestModule(BaseModule):
            def __init__(self):
                super().__init__()

            async def forward(self, **inputs):
                # Simulate processing that returns structured data
                name = inputs.get("name", "")
                age = inputs.get("age", 0)

                return Prediction(
                    outputs={
                        "greeting": f"Hello, {name}!",
                        "age_category": "adult" if age >= 18 else "minor",
                        "metadata": {"processed": True, "input_count": len(inputs)},
                    },
                    usage=Usage(),
                    success=True,
                )

        module = isolated_module(TestModule())

        # Add assertions to check the output structure
        assertions = [
            Assert.format(
                "output_structure",
                format_type="structure",
                required_keys=["greeting", "age_category", "metadata"],
            ),
        ]

        # Apply assertions to module
        assert_module_output(module, assertions, enable_backtracking=False)

        # Test valid inputs
        result = await module(name="Alice", age=25)

        assert result.success is True
        assert result.outputs["greeting"] == "Hello, Alice!"
        assert result.outputs["age_category"] == "adult"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_module_assertion_failure(self, isolated_module):
        """Test module assertion failure."""

        class FailingModule(BaseModule):
            async def forward(self, **inputs):
                return Prediction(
                    outputs={"result": 42},  # Wrong type for assertion
                    usage=Usage(),
                    success=True,
                )

        module = isolated_module(FailingModule())

        # Add assertion that will fail (require a key that doesn't exist)
        assertions = [
            Assert.format("missing_field", format_type="structure", required_keys=["missing_key"])
        ]
        assert_module_output(module, assertions, enable_backtracking=False)

        with pytest.raises(Exception) as exc_info:  # Could be AssertionError or ModuleError
            await module(input="test")

        assert "Missing required keys" in str(exc_info.value)


class TestBacktrackingWorkflow:
    """Test backtracking workflow with real scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_simple_backtracking_scenario(self):
        """Test simple backtracking scenario."""
        handler = BacktrackHandler(
            max_attempts=3,
            default_strategy=BacktrackStrategy.MODIFY_INPUTS,
        )

        # Mock forward function that succeeds on second try
        call_count = 0

        async def mock_forward(**inputs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call fails
                return Prediction(
                    outputs={"number": -5},  # Negative number
                    success=True,
                )
            else:
                # Second call succeeds
                return Prediction(
                    outputs={"number": 10},  # Positive number
                    success=True,
                )

        # Create assertion (not the result)
        assertion = Assert.value("positive_check", min_value=0)

        # Handle backtracking with the assertion object
        result = await handler.handle_failures(
            [assertion],  # Pass the assertion, not the result
            {"input_value": -5},
            mock_forward,
            original_value=-5,  # The value that originally failed
        )

        assert result.success is True
        assert result.outputs["number"] == 10
        assert call_count == 2  # Should have been called twice

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_backtracking_exhaustion(self):
        """Test backtracking when all attempts are exhausted."""
        handler = BacktrackHandler(max_attempts=2)

        # Mock forward function that always fails
        async def always_failing_forward(**inputs):
            return Prediction(
                outputs={"result": "invalid"},
                success=True,
            )

        assertion = Assert.value("type_check", expected_type=int)

        with pytest.raises(AssertionError) as exc_info:
            await handler.handle_failures(
                [assertion],  # Pass the assertion, not the result
                {"input": "test"},
                always_failing_forward,
                original_value="invalid",  # The value that originally failed
            )

        assert "All backtracking attempts exhausted" in str(exc_info.value)


class TestSuggestionGenerationWorkflow:
    """Test suggestion generation in real scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_comprehensive_suggestion_generation(self):
        """Test comprehensive suggestion generation."""
        generator = SuggestionGenerator()

        # Create multiple failed assertions
        failed_assertions = [
            Assert.value("type_check", expected_type=str).check(42),
            Assert.constraint("length_check", min_length=10).check("short"),
            Assert.format("json_check", format_type="json").check("invalid json"),
        ]

        suggestions = await generator.generate_suggestions(
            failed_assertions,
            original_value="test_input",
        )

        # Should contain suggestions from all failed assertions
        suggestion_text = " ".join(suggestions)
        assert "Convert value to str" in suggestion_text
        assert "Increase length to at least 10" in suggestion_text
        assert "valid JSON syntax" in suggestion_text
        assert "Address multiple validation issues" in suggestion_text


class TestCustomAssertionWorkflow:
    """Test custom assertion workflows."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_business_logic_validation(self):
        """Test custom assertions for business logic validation."""

        def is_valid_user_age(age):
            """Business rule: user age must be between 13 and 120."""
            return isinstance(age, int) and 13 <= age <= 120

        def suggest_valid_age(age):
            if not isinstance(age, int):
                return ["Age must be an integer"]
            elif age < 13:
                return ["Minimum age is 13 for registration"]
            elif age > 120:
                return ["Please verify age - maximum realistic age is 120"]
            return ["Use a valid age between 13 and 120"]

        async with assertion_context("user_validation") as context:
            age_assertion = Assert.custom(
                "valid_user_age",
                is_valid_user_age,
                suggest_valid_age,
                severity=AssertionSeverity.HARD,
            )
            context.add_assertion(age_assertion)

            # Test valid age
            results = context.check_all(25)
            assert results[0].passed is True

            # Test invalid age (too young)
            context.clear_results()
            results = context.check_all(10)
            assert results[0].passed is False
            assert "Minimum age is 13" in results[0].suggestions[0]

            # Test invalid age (too old)
            context.clear_results()
            results = context.check_all(150)
            assert results[0].passed is False
            assert "maximum realistic age is 120" in results[0].suggestions[0]


class TestComplexWorkflow:
    """Test complex workflow combining multiple features."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_user_registration_workflow(self, isolated_module):
        """Test complete user registration validation workflow."""

        class UserRegistrationModule(BaseModule):
            async def forward(self, **inputs):
                """Process user registration data."""
                user_data = {
                    "username": inputs.get("username", "").strip().lower(),
                    "email": inputs.get("email", "").strip().lower(),
                    "age": inputs.get("age", 0),
                    "profile": {
                        "created_at": "2023-01-01",
                        "active": True,
                        "preferences": inputs.get("preferences", {}),
                    },
                }

                return Prediction(
                    outputs=user_data,
                    usage=Usage(),
                    success=True,
                )

        module = isolated_module(UserRegistrationModule())

        # Define comprehensive validation assertions
        # Note: These assertions will be applied to the whole output dict
        assertions = [
            # Top-level structure validation (checking that the output has the right structure)
            Assert.format(
                "output_structure",
                format_type="structure",
                required_keys=["username", "email", "age", "profile"],
            ),
            # Profile nested structure validation - we'll check this separately
            Assert.custom(
                "profile_validation",
                lambda data: isinstance(data.get("profile"), dict)
                and all(k in data["profile"] for k in ["created_at", "active", "preferences"]),
            ),
            # Username validation - check on specific field
            Assert.custom(
                "username_validation",
                lambda data: (
                    isinstance(data.get("username"), str)
                    and len(data["username"]) >= 3
                    and len(data["username"]) <= 20
                    and data["username"].replace("_", "").isalnum()
                ),
            ),
            # Email validation
            Assert.custom(
                "email_validation",
                lambda data: "@" in data.get("email", "") and "." in data.get("email", ""),
            ),
            # Age validation
            Assert.custom(
                "valid_age",
                lambda data: isinstance(data.get("age"), int) and 13 <= data["age"] <= 120,
            ),
        ]

        # Apply assertions with backtracking
        assert_module_output(module, assertions, enable_backtracking=True)

        # Test valid registration
        result = await module(
            username="john_doe",
            email="john@example.com",
            age=25,
            preferences={"newsletter": True},
        )

        assert result.success is True
        assert result.outputs["username"] == "john_doe"
        assert result.outputs["email"] == "john@example.com"
        assert result.outputs["age"] == 25

        # Test invalid registration (should fail)
        from logillm.exceptions import ModuleError

        with pytest.raises(ModuleError):
            await module(
                username="a",  # Too short
                email="invalid-email",  # Invalid format
                age=5,  # Too young
            )


class TestSoftAssertionWorkflow:
    """Test soft assertion workflow that allows execution to continue."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_soft_assertions_continue_execution(self, isolated_module):
        """Test that soft assertions allow execution to continue."""

        class DataProcessingModule(BaseModule):
            async def forward(self, **inputs):
                data = inputs.get("data", "")
                processed = data.upper() if isinstance(data, str) else str(data)

                return Prediction(
                    outputs={
                        "processed_data": processed,
                        "warning_count": 0,
                    },
                    success=True,
                )

        module = isolated_module(DataProcessingModule())

        # Add soft assertions that will warn but not fail
        assertions = [
            Assert.value("data_type_warning", expected_type=str, severity=AssertionSeverity.SOFT),
            Assert.constraint(
                "data_length_warning", min_length=10, severity=AssertionSeverity.SOFT
            ),
        ]

        assert_module_output(module, assertions, enable_backtracking=False)

        # This should succeed despite soft assertion failures
        result = await module(data=42)  # Wrong type and too short

        assert result.success is True
        assert result.outputs["processed_data"] == "42"


class TestRealWorldScenarios:
    """Test real-world scenarios with assertions."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_api_response_validation(self):
        """Test API response validation scenario."""

        async def mock_api_call(**params):
            """Mock API call that returns structured data."""
            return Prediction(
                outputs={
                    "status": "success",
                    "data": {
                        "id": 123,
                        "name": "Test Item",
                        "price": 29.99,
                        "available": True,
                    },
                    "timestamp": "2023-01-01T12:00:00Z",
                },
                success=True,
            )

        # Define API response validation
        async with assertion_context("api_validation") as context:
            # Simulate API call first to get the response
            response = await mock_api_call(item_id=123)

            # Validate top-level structure
            top_level_assertions = [
                Assert.format(
                    "response_structure",
                    format_type="structure",
                    required_keys=["status", "data", "timestamp"],
                ),
                Assert.custom(
                    "status_check", lambda data: data.get("status") in {"success", "error"}
                ),
                Assert.custom(
                    "timestamp_check",
                    lambda data: isinstance(data.get("timestamp"), str)
                    and len(data["timestamp"]) > 0,
                ),
            ]

            for assertion in top_level_assertions:
                context.add_assertion(assertion)

            results = context.check_all(response.outputs)
            assert all(r.passed for r in results), (
                f"Failed validations: {[r.message for r in results if not r.passed]}"
            )

            # Now validate the nested data structure
            context.assertions.clear()
            context.results.clear()

            # Check structure of data field
            data_structure_assertion = Assert.format(
                "data_structure",
                format_type="structure",
                required_keys=["id", "name", "price", "available"],
            )
            context.add_assertion(data_structure_assertion)

            data_results = context.check_all(response.outputs["data"])
            assert all(r.passed for r in data_results), (
                f"Failed data structure validation: {[r.message for r in data_results if not r.passed]}"
            )

            # Check individual field types
            context.assertions.clear()
            context.results.clear()

            # Check id type
            id_assertion = Assert.value("id_type", expected_type=int)
            context.add_assertion(id_assertion)
            id_results = context.check_all(response.outputs["data"]["id"])
            assert all(r.passed for r in id_results), "Failed id type validation"

            # Check price type
            context.assertions.clear()
            context.results.clear()

            price_assertion = Assert.value("price_type", expected_type=float)
            context.add_assertion(price_assertion)
            price_results = context.check_all(response.outputs["data"]["price"])
            assert all(r.passed for r in price_results), "Failed price type validation"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_llm_output_validation(self):
        """Test LLM output validation scenario."""

        # Mock LLM response
        llm_response = """
        {
            "analysis": "The user query is asking about weather information",
            "intent": "weather_request",
            "entities": {
                "location": "New York",
                "date": "today"
            },
            "confidence": 0.95
        }
        """

        async with assertion_context("llm_validation") as context:
            # First validate that the string is valid JSON
            json_assertions = [
                Assert.format("json_format", format_type="json"),
            ]

            for assertion in json_assertions:
                context.add_assertion(assertion)

            # Check if the raw string is valid JSON
            json_results = context.check_all(llm_response.strip())
            assert all(r.passed for r in json_results), "Response is not valid JSON"

            # Parse and validate structure
            try:
                parsed_response = json.loads(llm_response.strip())

                # Clear context for structure validation
                context.assertions.clear()
                context.results.clear()

                structure_assertions = [
                    Assert.format(
                        "required_fields",
                        format_type="structure",
                        required_keys=["analysis", "intent", "entities", "confidence"],
                    ),
                ]

                for assertion in structure_assertions:
                    context.add_assertion(assertion)

                structure_results = context.check_all(parsed_response)
                assert all(r.passed for r in structure_results), "Missing required fields"

                # Validate specific fields
                context.assertions.clear()
                context.results.clear()

                # Check confidence value
                confidence_assertion = Assert.value(
                    "confidence_range", min_value=0.0, max_value=1.0
                )
                context.add_assertion(confidence_assertion)
                confidence_results = context.check_all(parsed_response.get("confidence", 0))
                assert all(r.passed for r in confidence_results), "Confidence out of range"

                # Check analysis length
                context.assertions.clear()
                context.results.clear()

                analysis_assertion = Assert.constraint("analysis_length", min_length=10)
                context.add_assertion(analysis_assertion)
                analysis_results = context.check_all(parsed_response.get("analysis", ""))
                assert all(r.passed for r in analysis_results), "Analysis too short"

            except json.JSONDecodeError:
                pytest.fail("LLM response is not valid JSON")


if __name__ == "__main__":
    pytest.main([__file__])

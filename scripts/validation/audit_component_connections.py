#!/usr/bin/env python3
"""Audit all component connections to find disconnections like the demo bug."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.modules import Parameter
from logillm.core.parameters import ParamDomain, ParamSpec, ParamType, SearchSpace
from logillm.core.predict import Predict
from logillm.optimizers.format_optimizer import FormatOptimizer, FormatOptimizerConfig, PromptFormat
from logillm.optimizers.hyperparameter import HyperparameterOptimizer
from logillm.providers import create_provider, register_provider


class ComponentConnectionAuditor:
    """Systematically audit component connections for silent failures."""

    def __init__(self):
        self.issues_found = []
        self.connections_tested = []

    def report_issue(
        self, component_a: str, component_b: str, issue: str, severity: str = "CRITICAL"
    ):
        """Report a connection issue."""
        self.issues_found.append(
            {"from": component_a, "to": component_b, "issue": issue, "severity": severity}
        )
        print(f"  ❌ {severity}: {component_a} → {component_b}")
        print(f"     Issue: {issue}")

    def report_success(self, component_a: str, component_b: str, test: str):
        """Report a successful connection."""
        self.connections_tested.append(
            {"from": component_a, "to": component_b, "test": test, "status": "PASS"}
        )
        print(f"  ✅ {component_a} → {component_b}: {test}")

    async def audit_hyperparameter_application(self):
        """Check if hyperparameter optimizer actually applies parameters."""
        print("\n🔍 AUDIT: Hyperparameter Optimizer → Module Parameters")
        print("-" * 60)

        # Setup provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Create module
        module = Predict("input -> output")

        # Check initial state
        initial_provider_temp = getattr(module.provider, "temperature", None)
        initial_config_temp = module.config.get("temperature", None)

        print(f"Initial provider temperature: {initial_provider_temp}")
        print(f"Initial config temperature: {initial_config_temp}")

        # Create hyperparameter optimizer
        search_space = SearchSpace(
            {
                "temperature": ParamSpec(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    domain=ParamDomain.GENERATION,
                    description="Temperature",
                    default=0.5,
                    range=(0.1, 0.9),
                    step=0.4,
                )
            }
        )

        optimizer = HyperparameterOptimizer(
            search_space=search_space,
            metric=lambda p, e: 1.0,  # Always succeed for testing
            strategy="grid",  # Correct parameter name
            n_trials=2,
        )

        # Optimize
        training_data = [{"inputs": {"input": "test"}, "outputs": {"output": "result"}}]
        result = await optimizer.optimize(module, training_data)
        optimized_module = result.optimized_module

        # Check if parameters were applied
        final_provider_temp = getattr(optimized_module.provider, "temperature", None)
        final_config_temp = optimized_module.config.get("temperature", None)
        final_params_temp = None
        if "temperature" in optimized_module.parameters:
            param = optimized_module.parameters["temperature"]
            if hasattr(param, "value"):
                final_params_temp = param.value

        print(f"Final provider temperature: {final_provider_temp}")
        print(f"Final config temperature: {final_config_temp}")
        print(f"Final parameters temperature: {final_params_temp}")

        # Check for disconnection
        temps_found = [
            t for t in [final_provider_temp, final_config_temp, final_params_temp] if t is not None
        ]

        if not temps_found:
            self.report_issue(
                "HyperparameterOptimizer",
                "Module",
                "Temperature not stored anywhere after optimization!",
                "CRITICAL",
            )
        elif len(set(temps_found)) > 1:
            self.report_issue(
                "HyperparameterOptimizer",
                "Module",
                f"Temperature stored inconsistently: provider={final_provider_temp}, config={final_config_temp}, params={final_params_temp}",
                "WARNING",
            )
        else:
            self.report_success(
                "HyperparameterOptimizer",
                "Module",
                f"Temperature correctly applied: {temps_found[0]}",
            )

        # Critical test: Does changing temperature actually affect API calls?
        if optimized_module.provider:
            # Make a test call and check what temperature is used
            # This would require mocking or instrumenting the provider
            pass

    async def audit_format_optimizer_application(self):
        """Check if format optimizer actually changes prompt formats."""
        print("\n🔍 AUDIT: Format Optimizer → Adapter Format")
        print("-" * 60)

        module = Predict("input -> output")
        initial_adapter_type = module.adapter.format_type.value
        print(f"Initial adapter format: {initial_adapter_type}")

        # Create format optimizer
        config = FormatOptimizerConfig(
            formats_to_test=[PromptFormat.JSON, PromptFormat.MARKDOWN],
            min_samples_per_format=1,
            max_samples_per_format=2,
        )

        optimizer = FormatOptimizer(metric=lambda p, e: 1.0, config=config)

        training_data = [{"inputs": {"input": "test"}, "outputs": {"output": "result"}}]
        result = await optimizer.optimize(module, training_data)
        optimized_module = result.optimized_module

        # Check if format was changed
        final_adapter_type = optimized_module.adapter.format_type.value
        print(f"Final adapter format: {final_adapter_type}")

        # Check where format is stored
        format_in_params = None
        if "prompt_format" in optimized_module.parameters:
            param = optimized_module.parameters["prompt_format"]
            if hasattr(param, "value"):
                format_in_params = param.value

        print(f"Format in parameters: {format_in_params}")

        if initial_adapter_type == final_adapter_type and format_in_params:
            self.report_issue(
                "FormatOptimizer",
                "Adapter",
                f"Format stored in parameters ({format_in_params}) but adapter still using {final_adapter_type}",
                "CRITICAL",
            )
        elif final_adapter_type != initial_adapter_type:
            self.report_success(
                "FormatOptimizer",
                "Adapter",
                f"Format changed from {initial_adapter_type} to {final_adapter_type}",
            )
        else:
            self.report_issue(
                "FormatOptimizer", "Adapter", "Format optimization had no effect", "WARNING"
            )

    async def audit_instruction_application(self):
        """Check if instruction modifications reach prompts."""
        print("\n🔍 AUDIT: Instruction Modifications → Actual Prompts")
        print("-" * 60)

        module = Predict("input -> output")

        # Check initial instructions
        initial_instructions = module.signature.instructions if module.signature else None
        print(f"Initial instructions: {initial_instructions}")

        # Manually modify instructions (simulating what an optimizer would do)
        if module.signature:
            module.signature.instructions = "MODIFIED: Always be helpful"

        # Also check if it's stored in parameters
        module.parameters["instructions"] = Parameter(
            value="PARAMETER: Be very helpful", learnable=True
        )

        # Check which instruction would be used
        final_instructions = module.signature.instructions if module.signature else None
        param_instructions = (
            module.parameters.get("instructions", {}).value
            if "instructions" in module.parameters
            else None
        )

        print(f"Final signature instructions: {final_instructions}")
        print(f"Parameter instructions: {param_instructions}")

        if final_instructions != param_instructions and param_instructions:
            self.report_issue(
                "InstructionOptimizer",
                "Signature",
                f"Instructions inconsistent: signature='{final_instructions}', params='{param_instructions}'",
                "CRITICAL",
            )
        elif final_instructions == "MODIFIED: Always be helpful":
            self.report_success(
                "InstructionOptimizer", "Signature", "Instructions correctly modified"
            )

    async def audit_provider_parameter_usage(self):
        """Check if provider actually uses configured parameters."""
        print("\n🔍 AUDIT: Module Config → Provider API Calls")
        print("-" * 60)

        from unittest.mock import AsyncMock, MagicMock

        # Create module with specific config
        module = Predict("input -> output")
        module.config = {"temperature": 0.3, "max_tokens": 100, "top_p": 0.9}

        # Mock provider to capture what parameters it receives
        mock_provider = AsyncMock()
        mock_provider.name = "test"
        mock_provider.model = "test-model"
        mock_provider.complete = AsyncMock(
            return_value=MagicMock(
                text="test response",
                usage=MagicMock(tokens=MagicMock(input_tokens=10, output_tokens=5)),
            )
        )

        module.provider = mock_provider

        # Make a call
        await module.forward(input="test")

        # Check what parameters were passed to provider
        if mock_provider.complete.called:
            call_kwargs = mock_provider.complete.call_args.kwargs
            print(f"Parameters passed to provider: {call_kwargs}")

            # Check if config parameters made it through
            if "temperature" in call_kwargs and call_kwargs["temperature"] == 0.3:
                self.report_success(
                    "Module.config", "Provider.complete", "Temperature correctly passed"
                )
            else:
                self.report_issue(
                    "Module.config",
                    "Provider.complete",
                    f"Temperature not passed or wrong value: {call_kwargs.get('temperature')}",
                    "CRITICAL",
                )
        else:
            self.report_issue(
                "Module", "Provider", "Provider.complete was never called!", "CRITICAL"
            )

    async def audit_all_connections(self):
        """Run all audits."""
        print("🔍 COMPREHENSIVE COMPONENT CONNECTION AUDIT")
        print("=" * 60)

        audits = [
            ("Hyperparameter Application", self.audit_hyperparameter_application),
            ("Format Optimizer Application", self.audit_format_optimizer_application),
            ("Instruction Application", self.audit_instruction_application),
            ("Provider Parameter Usage", self.audit_provider_parameter_usage),
        ]

        for audit_name, audit_func in audits:
            try:
                await audit_func()
            except Exception as e:
                print(f"\n❌ Audit '{audit_name}' failed with error: {e}")
                self.report_issue(audit_name, "Unknown", f"Audit crashed: {e}", "ERROR")

        # Summary
        print("\n" + "=" * 60)
        print("AUDIT SUMMARY")
        print("=" * 60)
        print(f"Connections tested: {len(self.connections_tested)}")
        print(f"Issues found: {len(self.issues_found)}")

        if self.issues_found:
            print("\n⚠️  CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for issue in self.issues_found:
                if issue["severity"] == "CRITICAL":
                    print(f"  - {issue['from']} → {issue['to']}: {issue['issue']}")

        return len(self.issues_found) == 0


async def main():
    """Run comprehensive audit."""
    auditor = ComponentConnectionAuditor()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  No OPENAI_API_KEY found - some audits may be limited")

    success = await auditor.audit_all_connections()

    if success:
        print("\n✅ All component connections verified!")
    else:
        print(f"\n❌ Found {len(auditor.issues_found)} connection issues that need fixing")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

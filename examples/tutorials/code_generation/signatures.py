"""
LogiLLM signatures for code generation tutorial.
"""

from logillm.core.signatures import InputField, OutputField, Signature


class AnalyzeLibraryDocs(Signature):
    """Analyze library documentation to extract key concepts and patterns."""

    library_name: str = InputField(desc="Name of the Python library")
    documentation_content: str = InputField(desc="Raw documentation content")

    core_concepts: str = OutputField(
        desc='Key concepts as JSON array: ["concept1", "concept2", "concept3"]'
    )
    main_classes: str = OutputField(
        desc='Important classes as JSON array: ["Class1: purpose", "Class2: purpose"]'
    )
    common_patterns: str = OutputField(
        desc='Usage patterns as JSON array: ["pattern1", "pattern2"]'
    )
    installation_method: str = OutputField(desc="How to install the library")
    import_statements: str = OutputField(
        desc='Import statements as JSON array: ["import library", "from library import module"]'
    )


class GenerateCodeExample(Signature):
    """Generate working code examples for a specific use case."""

    library_name: str = InputField()
    use_case: str = InputField(desc="Specific use case or functionality to demonstrate")
    core_concepts: str = InputField(desc="Key concepts from documentation analysis")
    common_patterns: str = InputField(desc="Common usage patterns")
    import_statements: str = InputField(desc="Relevant import statements")

    code_example: str = OutputField(desc="Complete, runnable code example")
    explanation: str = OutputField(desc="Step-by-step explanation of the code")
    best_practices: str = OutputField(desc='Best practices as JSON array: ["tip1", "tip2", "tip3"]')
    potential_issues: str = OutputField(desc='Common pitfalls as JSON array: ["issue1", "issue2"]')


class CreateLearningPlan(Signature):
    """Create a structured learning plan for mastering the library."""

    library_name: str = InputField()
    core_concepts: str = InputField()
    common_patterns: str = InputField()
    user_experience: str = InputField(
        desc="User's experience level (beginner, intermediate, advanced)"
    )

    learning_objectives: str = OutputField(
        desc='Learning goals as JSON array: ["objective1", "objective2"]'
    )
    suggested_examples: str = OutputField(
        desc='Progressive examples as JSON array: ["example1", "example2"]'
    )
    additional_resources: str = OutputField(
        desc='Resources as JSON array: ["resource1", "resource2"]'
    )
    time_estimate: str = OutputField(desc="Estimated time to become proficient")


class RefineCodeExample(Signature):
    """Refine and improve a code example based on feedback or requirements."""

    original_code: str = InputField(desc="Original code example")
    feedback: str = InputField(desc="User feedback or specific requirements")
    library_patterns: str = InputField(desc="Library-specific patterns to follow")

    improved_code: str = OutputField(desc="Refined and improved code example")
    changes_made: str = OutputField(desc='Improvements made as JSON array: ["change1", "change2"]')
    additional_features: str = OutputField(
        desc='Extension suggestions as JSON array: ["feature1", "feature2"]'
    )

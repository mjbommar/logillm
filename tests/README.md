# LogiLLM Tests

This directory contains comprehensive tests organized into two separate folders:

## 📁 Unit Tests (`tests/unit/`)

**Purpose**: Fast, isolated tests using mock objects  
**Rule**: MUST use mocks, NEVER real API calls  
**Run frequency**: Every commit, CI always runs these

```bash
# Run unit tests only
uv run pytest tests/unit/

# Run unit tests with coverage
uv run pytest tests/unit/ --cov=logillm
```

### Structure:
- `tests/unit/core/` - Core module tests (Predict, signatures, parameters, etc.)
- `tests/unit/optimizers/` - Optimizer tests with mock components  
- `tests/unit/providers/` - Provider interface and MockProvider tests
- `tests/unit/fixtures/` - Shared mock components and fixtures
- `tests/unit/conftest.py` - Unit test configuration and fixtures

## 📁 Integration Tests (`tests/integration/`)

**Purpose**: End-to-end tests with real API calls  
**Rule**: NEVER use mocks, ALWAYS real APIs  
**Run frequency**: On demand, CI only with API keys

```bash
# Run integration tests (requires OPENAI_API_KEY)
uv run pytest tests/integration/ -m integration

# Run specific provider integration tests
uv run pytest tests/integration/providers/ -m openai
```

### Structure:
- `tests/integration/providers/` - Real provider API tests
- `tests/integration/examples/` - Validate all examples/ actually work
- `tests/integration/optimization/` - End-to-end optimization workflows  
- `tests/integration/workflows/` - Complete user workflows and benchmarks
- `tests/integration/conftest.py` - Integration test configuration

## 🏷️ Test Markers

Tests are marked for selective execution:

- `@pytest.mark.unit` - Fast unit tests with mocks
- `@pytest.mark.integration` - Slow integration tests with real APIs  
- `@pytest.mark.openai` - Tests requiring OpenAI API key
- `@pytest.mark.slow` - Tests that take > 5 seconds

## 🚀 Running Tests

```bash
# Default: Unit tests only (fast)
uv run pytest

# All unit tests explicitly  
uv run pytest tests/unit/

# Integration tests (requires API keys)
uv run pytest tests/integration/ -m integration

# Run everything (unit + integration)
uv run pytest tests/ -m "not slow"

# Just OpenAI integration tests
uv run pytest -m "integration and openai"

# Skip integration tests
uv run pytest -m "not integration"
```

## 🔧 Configuration

- **pytest.ini**: Configures markers and default behavior
- **conftest.py files**: Provide fixtures for each test type
- **Default behavior**: Only runs unit tests (fast feedback)

## 📊 Test Coverage

Unit tests focus on:
- ✅ Logic and behavior verification
- ✅ Error handling and edge cases  
- ✅ Interface compliance
- ✅ Parameter validation

Integration tests focus on:
- ✅ Real API functionality
- ✅ End-to-end workflows
- ✅ Example validation  
- ✅ Performance benchmarks

## 🔑 API Keys

Integration tests require environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
# Add other provider keys as needed
```

## 📈 CI/CD Usage

**Recommended CI setup**:
```yaml
# Always run unit tests
- run: uv run pytest tests/unit/

# Only run integration tests if API keys available
- if: env.OPENAI_API_KEY
  run: uv run pytest tests/integration/ -m integration
```

This two-folder approach ensures:
1. **Fast feedback** from unit tests (no API delays)
2. **Real validation** from integration tests (actual API behavior)  
3. **Clear separation** of concerns and responsibilities
4. **Scalable testing** as the codebase grows
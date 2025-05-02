# AnimateDiff Endpoint Tests

This directory contains tests for the AnimateDiff Endpoint project.

## Test Files

- `test_check_data_format.py`: Tests for the `check_data_format` function in `inference_util.py`. This function validates the input data format for the AnimateDiff API.
- `mock_inference_util.py`: A mock version of the `inference_util.py` file that contains only the `check_data_format` function, without dependencies on PyTorch and other libraries. This allows running the tests without installing all the dependencies.

## Running Tests

To run the tests, navigate to the `tests` directory and run:

```bash
python -m unittest test_check_data_format.py
```

## Adding New Tests

When adding new tests, consider the following:

1. If the test requires heavy dependencies like PyTorch, consider creating a mock version of the function to test.
2. Add the test file to this directory with a descriptive name.
3. Update this README.md file to document the new test.

## Test Coverage

The current test coverage includes:

- Input validation for the AnimateDiff API
- Validation of parameter types and formats

Future tests could include:

- Testing the AnimateDiff class methods
- Testing the server endpoints
- Integration tests for the full pipeline
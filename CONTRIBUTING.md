# Contributing to Multimodal Inference Service

Thank you for your interest in contributing to this project! This guide outlines how to contribute effectively.

## Development Setup

### Prerequisites
- Python 3.12+
- Docker (optional)
- Git

### Local Development
1. Clone the repository:
   ```bash
   git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-10-multimodal-inference-service.git
   cd playbook-10-multimodal-inference-service
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the data and run the tests:
   ```bash
   python download_sample_data.py
   make smoke    # or: pytest -q
   ```

## Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and modular
- No emojis in code or documentation

## Testing

Before submitting a pull request:

1. Run the smoke tests:
   ```bash
   make smoke
   ```

2. Test service startup:
   ```bash
   uvicorn service:app --host 0.0.0.0 --port 8000
   ```

3. Test Docker build:
   ```bash
   docker build -t multimodal-inference .
   ```

4. Run canary simulation:
   ```bash
   python simulate_canary.py --max-samples 400 --canary-traffic 20
   ```

## Contributing Guidelines

### Bug Reports
- Use the issue tracker to report bugs
- Include Python version, OS, and error messages
- Provide minimal reproduction steps

### Feature Requests
- Describe the use case and expected behavior
- Consider backward compatibility
- Discuss architectural impacts

### Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push and create a pull request

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Keep the first line under 50 characters
- Provide context in the body if needed

Example:
```
Add support for multiple camera views

- Process wrist_rgb and overhead_rgb directories
- Update find_latest_image to handle camera selection
- Add camera parameter to inference endpoint
```

## Architecture Guidelines

### Model Components
- Keep vision and telemetry processing separate
- Maintain compatibility with RLBench data format
- Use robust error handling for data loading

### Service Design
- Follow REST API best practices
- Implement proper logging
- Add health checks for monitoring

### Deployment
- Maintain Docker compatibility
- Keep manifest-based configuration
- Support canary deployment patterns

## Documentation

- Update README.md for new features
- Document API changes
- Add inline comments for complex logic
- Keep examples up to date

## Questions?

- Open an issue for questions
- Check existing issues and documentation first
- Provide context about your use case

Thank you for contributing!
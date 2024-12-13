# Install dependencies
```bash
python -m pip venv .venv
source ./venv/bin/activate
pip install poetry
poetry install
```

# Build
```bash
pip install -q build
python -m build

pip install dist/my_package-0.0.1-py3-none-any.whl
pip install dist/my_package-0.0.1.tar

```
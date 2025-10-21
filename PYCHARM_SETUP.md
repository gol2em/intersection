# PyCharm Configuration for UV

This guide shows how to configure PyCharm to use `uv run` by default.

## Method 1: Configure Python Interpreter to Use UV's Virtual Environment

1. **Open PyCharm Settings**
   - Windows/Linux: `File` → `Settings`
   - macOS: `PyCharm` → `Preferences`

2. **Navigate to Python Interpreter**
   - Go to `Project: intersection` → `Python Interpreter`

3. **Add UV's Virtual Environment**
   - Click the gear icon ⚙️ next to the Python Interpreter dropdown
   - Select `Add...`
   - Choose `Existing Environment`
   - Click the `...` button to browse
   - Navigate to: `D:\lsec\Python\Intersection\.venv\Scripts\python.exe`
   - Click `OK`

4. **Verify Installation**
   - PyCharm should now show all packages installed via `uv`
   - You can run scripts normally and they'll use the uv environment

## Method 2: Configure Run Configurations to Use UV

### For Individual Python Files

1. **Open Run Configuration**
   - Right-click on a Python file (e.g., `examples/example_2d.py`)
   - Select `Modify Run Configuration...`

2. **Edit Configuration**
   - In the `Script path` field, keep your Python file path
   - OR change to use `uv run`:
     - Change configuration type to `Shell Script`
     - Command: `uv run python examples/example_2d.py`
     - Working directory: `D:\lsec\Python\Intersection`

### For All Python Files (Template)

1. **Edit Run Configuration Templates**
   - Go to `Run` → `Edit Configurations...`
   - Click `Edit configuration templates...` (bottom left)
   - Select `Python`

2. **Modify Template**
   - You can't directly use `uv run` in the template, but you can:
   - Set the Python interpreter to the `.venv` interpreter (Method 1)
   - This ensures all new run configurations use the correct environment

## Method 3: Use External Tools

1. **Create External Tool for UV Run**
   - Go to `Settings` → `Tools` → `External Tools`
   - Click `+` to add a new tool
   - Configure:
     - **Name**: `UV Run`
     - **Program**: `uv` (or full path if not in PATH)
     - **Arguments**: `run python $FilePath$`
     - **Working directory**: `$ProjectFileDir$`

2. **Use the Tool**
   - Right-click any Python file
   - Select `External Tools` → `UV Run`

## Method 4: Terminal Integration (Recommended for Development)

1. **Configure Terminal to Auto-activate UV Environment**
   - Go to `Settings` → `Tools` → `Terminal`
   - In `Shell path`, you can add activation script
   - For PowerShell: Keep default, then manually run:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

2. **Run Commands in Terminal**
   - Open PyCharm's integrated terminal (Alt+F12)
   - Run: `uv run python examples/example_2d.py`
   - Or activate venv first: `.venv\Scripts\Activate.ps1`
   - Then run normally: `python examples/example_2d.py`

## Method 5: Use UV Directly in PATH

If `uv` is installed globally and in your PATH:

1. **Verify UV in PATH**
   ```powershell
   uv --version
   ```

2. **Create a Run Configuration**
   - Go to `Run` → `Edit Configurations...`
   - Click `+` → `Python`
   - Name: `Example 2D (UV)`
   - Script path: `examples/example_2d.py`
   - Python interpreter: Select the `.venv` interpreter
   - Click `OK`

## Recommended Setup

**Best approach for this project:**

1. ✅ **Use Method 1** (Configure interpreter to use `.venv`)
   - This is the simplest and most integrated with PyCharm
   - All PyCharm features (debugging, code completion, etc.) work normally

2. ✅ **Use Method 4** (Terminal) for package management
   - Use `uv add package-name` in the terminal
   - Use `uv sync` to install dependencies
   - Run scripts with `uv run python script.py` when needed

## Installing Dependencies with UV

Once configured, install project dependencies:

```powershell
# In PyCharm terminal or external terminal
cd D:\lsec\Python\Intersection

# Sync all dependencies from pyproject.toml
uv sync

# Or install specific packages
uv pip install numpy scipy matplotlib

# Install dev dependencies
uv sync --all-extras
```

## Running Examples

### From PyCharm UI
1. Open `examples/example_2d.py`
2. Right-click in the editor
3. Select `Run 'example_2d'`

### From Terminal
```powershell
# Using uv run
uv run python examples/example_2d.py

# Or activate venv first
.venv\Scripts\Activate.ps1
python examples/example_2d.py
```

## Running Tests

### From PyCharm UI
1. Right-click on `tests` folder
2. Select `Run 'pytest in tests'`

### From Terminal
```powershell
# Using uv
uv run pytest

# Or with activated venv
.venv\Scripts\Activate.ps1
pytest
```

## Troubleshooting

### UV not found
- Make sure `uv` is installed: `pip install uv`
- Or install globally following UV installation instructions
- Add UV to PATH if installed via standalone installer

### Virtual environment not recognized
- Delete `.venv` folder
- Run `uv venv` to recreate
- Reconfigure PyCharm interpreter

### Import errors
- Make sure dependencies are installed: `uv sync`
- Check that PyCharm is using the correct interpreter
- Invalidate caches: `File` → `Invalidate Caches / Restart`

## Additional Resources

- UV Documentation: https://github.com/astral-sh/uv
- PyCharm Python Interpreter Guide: https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html


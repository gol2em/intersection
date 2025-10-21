# Project Synchronization Guide

## Quick Start: Sync with Another Computer

### Method 1: Using GitHub (Recommended)

#### First Time Setup (Computer A)

1. **Create GitHub repository**
   - Go to https://github.com/new
   - Repository name: `Intersection`
   - Keep it private or public (your choice)
   - Don't initialize with README

2. **Commit and push your code**
   ```bash
   cd D:/lsec/Python/Intersection
   
   # Add all files
   git add -A
   
   # Commit
   git commit -m "Initial commit: n-dimensional intersection framework"
   
   # Add remote (replace YOUR_USERNAME with your GitHub username)
   git remote add origin https://github.com/YOUR_USERNAME/Intersection.git
   
   # Push to GitHub
   git push -u origin master
   ```

#### Setup on Computer B

1. **Clone the repository**
   ```bash
   # Navigate to where you want the project
   cd /path/to/your/projects
   
   # Clone from GitHub
   git clone https://github.com/YOUR_USERNAME/Intersection.git
   cd Intersection
   ```

2. **Install dependencies**
   ```bash
   # Install uv if not already installed
   pip install uv
   
   # Create virtual environment and install dependencies
   uv sync
   ```

3. **Verify installation**
   ```bash
   # Activate virtual environment
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   
   # Run tests
   pytest tests/ -v
   ```

#### Daily Workflow

**On Computer A (after making changes):**
```bash
# Check what changed
git status

# Add changes
git add -A

# Commit with descriptive message
git commit -m "Add new feature X"

# Push to GitHub
git push
```

**On Computer B (to get latest changes):**
```bash
# Pull latest changes
git pull

# Update dependencies if pyproject.toml changed
uv sync

# Run tests to verify
pytest tests/ -v
```

---

### Method 2: Using GitLab

Same as GitHub, but use GitLab:
1. Go to https://gitlab.com
2. Create new project
3. Follow same steps as GitHub

---

### Method 3: Using Git with SSH (More Secure)

#### Setup SSH Key (One-time)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub
```

Add the public key to GitHub:
- GitHub → Settings → SSH and GPG keys → New SSH key

#### Use SSH URL instead of HTTPS

```bash
git remote add origin git@github.com:YOUR_USERNAME/Intersection.git
git push -u origin master
```

---

### Method 4: Manual Sync (Not Recommended)

If you can't use git, use cloud storage:

#### Files to sync:
- ✅ `src/` - Source code
- ✅ `tests/` - Tests
- ✅ `examples/` - Examples
- ✅ `pyproject.toml` - Dependencies
- ✅ `*.md` - Documentation

#### Files to SKIP:
- ❌ `.venv/` - Virtual environment (recreate)
- ❌ `__pycache__/` - Python cache
- ❌ `.idea/` - IDE settings
- ❌ `*.pyc` - Compiled files
- ❌ `*_output.txt` - Test outputs
- ❌ `*_old.py` - Backup files

#### On the other computer:
```bash
# Copy files to new location
# Then install dependencies
uv sync
```

---

## Troubleshooting

### Problem: "git push" asks for username/password repeatedly

**Solution:** Use SSH keys (see Method 3) or configure credential helper:
```bash
git config --global credential.helper store
```

### Problem: Merge conflicts

**Solution:**
```bash
# Pull with rebase
git pull --rebase

# If conflicts occur, resolve them manually
# Then continue
git rebase --continue
```

### Problem: Different Python versions on computers

**Solution:** Specify Python version in `pyproject.toml`:
```toml
[project]
requires-python = ">=3.9"
```

### Problem: Dependencies not syncing

**Solution:**
```bash
# Remove lock file and resync
rm uv.lock
uv sync
```

---

## Best Practices

### 1. Commit Often
```bash
# Good: Small, focused commits
git commit -m "Fix interpolation bug for degree 10"
git commit -m "Add tests for sphere"

# Bad: Large, vague commits
git commit -m "Various changes"
```

### 2. Pull Before Push
```bash
# Always pull before starting work
git pull

# Make changes...

# Then push
git push
```

### 3. Use Branches for Experiments
```bash
# Create branch for new feature
git checkout -b feature/new-solver

# Work on feature...

# Merge when ready
git checkout master
git merge feature/new-solver
```

### 4. Don't Commit Generated Files
Already handled by `.gitignore`:
- Virtual environments
- Cache files
- Test outputs
- IDE settings

### 5. Keep Dependencies Updated
```bash
# Update dependencies
uv sync --upgrade

# Commit updated lock file
git add uv.lock
git commit -m "Update dependencies"
```

---

## Quick Reference

### Common Git Commands

```bash
# Check status
git status

# Add all changes
git add -A

# Commit
git commit -m "Message"

# Push
git push

# Pull
git pull

# View history
git log --oneline

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all local changes
git reset --hard HEAD
```

### Common UV Commands

```bash
# Install dependencies
uv sync

# Add new package
uv add package-name

# Remove package
uv remove package-name

# Update all packages
uv sync --upgrade

# Run script
uv run python script.py
```

---

## Summary

**Recommended Setup:**
1. ✅ Use GitHub or GitLab
2. ✅ Use SSH keys for authentication
3. ✅ Commit often with clear messages
4. ✅ Pull before starting work
5. ✅ Use `uv sync` to keep dependencies in sync

**On Computer A:**
```bash
git add -A
git commit -m "Description"
git push
```

**On Computer B:**
```bash
git pull
uv sync
```

That's it! Your project stays synchronized across computers.


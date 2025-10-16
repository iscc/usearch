# Building and Using Python Wheels from GitHub Releases

This fork of USearch publishes prebuilt binary Python wheels to GitHub Releases instead of PyPI. This allows you
to use the patched version in your projects without publishing to the public package index.

## For Wheel Consumers

### Quick Install with UV

To install a specific wheel directly:

```bash
# Replace {VERSION}, {PYTHON}, {ABI}, and {PLATFORM} with appropriate values
uv pip install https://github.com/iscc/usearch/releases/download/v2.21.1-patched/usearch-2.21.1-cp311-cp311-manylinux_2_28_x86_64.whl
```

### Finding the Right Wheel

1. Visit the [Releases page](https://github.com/iscc/usearch/releases)
2. Choose the release version you need
3. Find the wheel matching your platform:
    - **Python version**: `cp38`, `cp39`, `cp310`, `cp311`, `cp312`, `cp313`, `cp313t` (or `pp37`-`pp310` for
      PyPy)
    - **Platform**:
        - Linux: `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `musllinux_1_2_x86_64`,
          `musllinux_1_2_aarch64`
        - macOS: `macosx_*_x86_64`, `macosx_*_arm64`, `macosx_*_universal2`
        - Windows: `win_amd64`, `win_arm64`

### Example Wheel Filenames

```
usearch-2.21.1-cp311-cp311-manylinux_2_28_x86_64.whl     # Linux x86_64, Python 3.11
usearch-2.21.1-cp312-cp312-macosx_11_0_arm64.whl        # macOS ARM64, Python 3.12
usearch-2.21.1-cp310-cp310-win_amd64.whl                # Windows x64, Python 3.10
usearch-2.21.1-cp313t-cp313t-manylinux_2_28_x86_64.whl  # Linux x86_64, Python 3.13 (no-GIL)
```

### Using in pyproject.toml

#### Option 1: Direct URL Dependency (Platform-Specific)

For a specific platform:

```toml
[project]
dependencies = [
    "usearch @ https://github.com/iscc/usearch/releases/download/v2.21.1-patched/usearch-2.21.1-cp311-cp311-manylinux_2_28_x86_64.whl",
]
```

**Limitation**: This locks you to a specific platform and Python version. Not suitable for cross-platform
projects.

#### Option 2: Use uv's Lock File (Recommended)

Let uv resolve the correct wheel for each platform:

1. Add to `pyproject.toml`:
   ```toml
   [project]
   dependencies = [
       "usearch",
   ]

   [tool.uv.sources]
   usearch = { url = "https://github.com/iscc/usearch/releases/download/v2.21.1-patched/usearch-2.21.1-cp311-cp311-manylinux_2_28_x86_64.whl" }
   ```

2. Or use `uv add` with `--index-url`:
   ```bash
   # First add the dependency manually to pyproject.toml, then:
   uv sync
   ```

#### Option 3: Multiple Platform URLs (Advanced)

For multi-platform support, you can use platform markers:

```toml
[project]
dependencies = [
    "usearch @ https://github.com/iscc/usearch/releases/download/v2.21.1-patched/usearch-2.21.1-cp311-cp311-manylinux_2_28_x86_64.whl ; sys_platform == 'linux' and platform_machine == 'x86_64'",
    "usearch @ https://github.com/iscc/usearch/releases/download/v2.21.1-patched/usearch-2.21.1-cp311-cp311-macosx_11_0_arm64.whl ; sys_platform == 'darwin' and platform_machine == 'arm64'",
    "usearch @ https://github.com/iscc/usearch/releases/download/v2.21.1-patched/usearch-2.21.1-cp311-cp311-win_amd64.whl ; sys_platform == 'win32'",
]
```

**Note**: This is verbose and requires updating URLs for each platform.

### Using with pip

While this project is designed for UV, you can also use pip:

```bash
pip install https://github.com/iscc/usearch/releases/download/v2.21.1-patched/usearch-2.21.1-cp311-cp311-manylinux_2_28_x86_64.whl
```

### Using in requirements.txt

```
https://github.com/iscc/usearch/releases/download/v2.21.1-patched/usearch-2.21.1-cp311-cp311-manylinux_2_28_x86_64.whl
```

## For Wheel Builders (Maintainers)

### Building Wheels Locally

Build for your current platform:

```bash
# Install build tools
uv pip install cibuildwheel

# Build wheels
cibuildwheel --output-dir wheelhouse

# Wheels will be in ./wheelhouse/
```

### Creating a Release

1. **Ensure all changes are committed**
   ```bash
   git add -A
   git commit -m "feat: your changes here"
   ```

2. **Create and push a version tag**
   ```bash
   # Tag format: v{VERSION}-patched (or any v* pattern)
   git tag v2.21.1-patched
   git push origin v2.21.1-patched
   ```

3. **Monitor the build**
    - Go to [Actions](https://github.com/iscc/usearch/actions)
    - Watch the "Build Python Wheels" workflow
    - Wait for all platform builds to complete (~20-30 minutes)

4. **Verify the release**
    - Check [Releases](https://github.com/iscc/usearch/releases)
    - Confirm all expected wheels are present
    - Test installation from the release

### Manual Release Creation

If you prefer manual control:

```bash
# Trigger workflow manually from GitHub Actions UI
# Or use gh CLI:
gh workflow run build-wheels.yml -f tag=v2.21.1-patched
```

### Expected Wheel Count

For a full release, expect approximately **77 wheels**:

- 7 Python versions × 3 OS types × multiple architectures
- CPython: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.13t
- PyPy: 3.7, 3.8, 3.9, 3.10
- Platforms: Linux (x86_64, aarch64), macOS (x86_64, arm64, universal2), Windows (AMD64, ARM64)

### Troubleshooting Builds

**Build fails on specific platform:**

- Check the Actions log for that platform
- Common issues: missing dependencies, compiler errors, test failures

**Wheels not appearing in release:**

- Ensure GitHub Actions has `contents: write` permission
- Check if the tag matches the pattern `v*`

**Testing a wheel before release:**

```bash
# Download artifact from Actions run
# Or build locally and test:
uv pip install ./wheelhouse/usearch-*.whl
python -c "import usearch; print(usearch.__version__)"
```

## Version Numbering

This fork uses the following versioning scheme:

- Base version follows upstream (e.g., `2.21.1`)
- Tag includes `-patched` suffix (e.g., `v2.21.1-patched`)
- For multiple patches to the same upstream version, use `-patch.N` (e.g., `v2.21.1-patch.2`)

## Differences from Upstream

This fork includes patches that are not in the upstream USearch repository. To see what's different:

```bash
# View commits since fork
git log --oneline upstream/main..main

# View specific patches
git diff upstream/main..main
```

## License

This fork maintains the same Apache-2.0 license as the upstream USearch project.

## Acknowledgments

This is a fork of [USearch by Unum](https://github.com/unum-cloud/usearch). All credit for the original
implementation goes to the Unum team and contributors.

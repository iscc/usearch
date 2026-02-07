# Installing usearch-iscc

This fork of USearch is published on PyPI as
[`usearch-iscc`](https://pypi.org/project/usearch-iscc/). The Python import name remains `usearch` for
compatibility with the upstream API.

## Installation

### From PyPI (Recommended)

```bash
pip install usearch-iscc
```

Or with uv:

```bash
uv pip install usearch-iscc
```

### Using in pyproject.toml

```toml
[project]
dependencies = [
    "usearch-iscc>=2.23.4",
]
```

### From GitHub Pages Index (Alternative)

A PEP 503 compliant package index is also available on GitHub Pages:

```bash
pip install usearch-iscc --extra-index-url https://iscc.github.io/usearch/
```

Or in `pyproject.toml` with uv:

```toml
[[tool.uv.index]]
url = "https://iscc.github.io/usearch/"
```

## Available Platforms

Pre-built wheels are available for:

- **Python versions:** 3.10, 3.11, 3.12, 3.13, 3.14
- **Operating systems:**
    - Linux: x86_64, aarch64 (manylinux_2_28, musllinux_1_2)
    - macOS: x86_64, arm64, universal2
    - Windows: AMD64, ARM64

## For Maintainers

### Building and Releasing

The build and deployment process is fully automated:

1. **Make your changes** and commit them to the `main` branch

2. **Create and push a version tag:**
   ```bash
   git tag v2.23.4
   git push origin v2.23.4
   ```

3. **GitHub Actions automatically:**
    - Builds wheels for all supported platforms via cibuildwheel
    - Publishes wheels to PyPI
    - Generates a PEP 503 compliant index on GitHub Pages

4. **Monitor the build:**
    - Visit: https://github.com/iscc/usearch/actions
    - Workflow: "Build and Publish Wheels to GitHub Pages"

5. **Verify deployment:**
    - PyPI: https://pypi.org/project/usearch-iscc/
    - GitHub Pages: https://iscc.github.io/usearch/

### Build Configuration

The workflow builds wheels with:

- **Python versions:** 3.10, 3.11, 3.12, 3.13, 3.14 (no PyPy)
- **Platforms:** Linux (manylinux_2_28, musllinux_1_2), macOS, Windows
- **Architectures:** x86_64, aarch64/arm64, universal2 (macOS)

### Testing Locally

Build wheels locally before releasing:

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build for your current platform
cibuildwheel --output-dir wheelhouse

# Test the wheel
pip install wheelhouse/*.whl
python -c "import usearch; print(usearch.__version__)"
```

### Manually Triggering Builds

If needed, you can retag to trigger a rebuild:

```bash
git tag -d v2.23.4
git push origin :refs/tags/v2.23.4
git tag v2.23.4
git push origin v2.23.4
```

## Version Numbering

This fork follows the upstream version scheme (e.g., `2.23.4`). Tags use the format `v2.23.4`.

## Differences from Upstream

This fork includes patches not in the upstream USearch repository. To see what's different:

```bash
# Add upstream remote if not already added
git remote add upstream https://github.com/unum-cloud/usearch.git
git fetch upstream

# View commits since fork
git log --oneline upstream/main..main

# View full diff
git diff upstream/main..main
```

## License

This fork maintains the same Apache-2.0 license as the upstream USearch project.

## Acknowledgments

This is a fork of [USearch by Unum](https://github.com/unum-cloud/usearch). All credit for the original
implementation goes to the Unum team and contributors.

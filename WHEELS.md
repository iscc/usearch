# Using USearch from GitHub Pages Python Index

This fork of USearch publishes prebuilt binary Python wheels to a GitHub Pages hosted package index. This allows
you to use the patched version in your projects without any platform-specific configuration.

## For Package Users

### Quick Install with UV

Simply add our package index to your project and install like any other package:

```bash
uv pip install usearch --index-url https://iscc.github.io/usearch/
```

### Using in pyproject.toml (Recommended)

Add to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "usearch",  # No platform-specific URL needed!
]

[[tool.uv.index]]
url = "https://iscc.github.io/usearch/"
```

Then install:

```bash
uv sync
```

**UV automatically selects the correct wheel for your platform!**

### Using with pip

```bash
pip install usearch --index-url https://iscc.github.io/usearch/
```

Or in `requirements.txt`:

```
--index-url https://iscc.github.io/usearch/
usearch
```

### Combining with PyPI

If your project uses both public PyPI packages and our patched usearch:

```toml
[project]
dependencies = [
    "usearch",
    "numpy",
    "scipy",
]

[tool.uv]
index-url = "https://pypi.org/simple"

[[tool.uv.index]]
url = "https://iscc.github.io/usearch/"
```

UV will look for packages in PyPI first, then fall back to our index for `usearch`.

## Available Platforms

Pre-built wheels are available for:

- **Python versions:** 3.10, 3.11, 3.12, 3.13
- **Operating systems:**
    - Linux (x86_64) - manylinux2014
    - macOS (x86_64)
    - Windows (x86_64)

## How It Works

When you add our GitHub Pages URL as a package index:

1. UV queries the index for available `usearch` packages
2. Finds all wheels for different platforms and Python versions
3. Automatically downloads the correct wheel for your system
4. Installs it like any PyPI package

**No platform detection needed in your configuration!**

## For Maintainers

### Building and Releasing

The build and deployment process is fully automated:

1. **Make your changes** and commit them to the `main` branch

2. **Create and push a version tag:**
   ```bash
   git tag v2.21.1-patched
   git push origin v2.21.1-patched
   ```

3. **GitHub Actions automatically:**
    - Builds wheels for all supported platforms (12 wheels total)
    - Generates a PEP 503 compliant index
    - Deploys to GitHub Pages at `https://iscc.github.io/usearch/`

4. **Monitor the build:**
    - Visit: https://github.com/iscc/usearch/actions
    - Workflow: "Build and Publish Wheels to GitHub Pages"
    - Duration: ~15-20 minutes

5. **Verify deployment:**
    - Check: https://iscc.github.io/usearch/
    - Browse to: https://iscc.github.io/usearch/usearch/ to see all wheels

### Build Configuration

The workflow builds wheels with:

- **Python versions:** 3.10, 3.11, 3.12, 3.13 (no PyPy)
- **Architecture:** x86_64 only (no ARM builds for speed)
- **Platforms:** Linux (manylinux2014), macOS, Windows
- **Optimizations:**
    - Parallel builds across all platforms
    - GitHub Actions cache for pip packages
    - Skip tests during wheel building (run tests separately)

### Expected Wheel Count

For each release, expect **12 wheels**:
- 4 Python versions × 3 platforms = 12 wheels

Example:
```
usearch-2.21.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
usearch-2.21.0-cp310-cp310-macosx_10_9_x86_64.whl
usearch-2.21.0-cp310-cp310-win_amd64.whl
... (9 more)
```

### Troubleshooting Builds

**Build fails on specific platform:**
- Check the Actions log for detailed error messages
- Common issues: missing dependencies, compilation errors
- Test locally with: `cibuildwheel --platform linux`

**GitHub Pages not updating:**
- Verify GitHub Pages is enabled in repository Settings → Pages
- Source should be: "GitHub Actions"
- Check the "deploy_pages" job completed successfully
- Pages updates can take 1-2 minutes to propagate

**Wheels not being found by UV:**
- Verify the index structure at https://iscc.github.io/usearch/usearch/
- Check that `index.html` exists and lists wheels
- Ensure wheel filenames follow PEP 427 naming convention

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

If needed, you can manually trigger the workflow:

```bash
# Retag to trigger rebuild
git tag -d v2.21.0-patched
git push origin :refs/tags/v2.21.0-patched
git tag v2.21.0-patched
git push origin v2.21.0-patched
```

## Version Numbering

This fork follows this versioning scheme:

- **Base version:** Matches upstream (e.g., `2.21.0`)
- **Tag suffix:** `-patched` to indicate fork (e.g., `v2.21.0-patched`)
- **Multiple patches:** Use `-patch.N` for iterations (e.g., `v2.21.0-patch.2`)

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

## Why GitHub Pages Instead of PyPI?

- ✅ No PyPI account needed
- ✅ No name conflicts with official package
- ✅ Full version control through Git
- ✅ Free hosting
- ✅ Works seamlessly with UV and pip
- ✅ Can include experimental patches without affecting upstream
- ✅ Fast deployment (no review process)

## License

This fork maintains the same Apache-2.0 license as the upstream USearch project.

## Acknowledgments

This is a fork of [USearch by Unum](https://github.com/unum-cloud/usearch). All credit for the original
implementation goes to the Unum team and contributors.

# GitHub Actions for PyPI Publishing

This repository is configured to automatically publish releases to PyPI using GitHub Actions.

## How it works

1. When you create a new release on GitHub, the workflow in `.github/workflows/publish.yml` is triggered
2. The workflow builds the package using the standard Python build tools
3. The package is then uploaded to PyPI using twine

## Setting up PyPI token

To enable automatic publishing, you need to set up a PyPI API token:

1. Go to https://pypi.org/manage/account/ and create an API token
2. In your GitHub repository settings, go to "Secrets and variables" → "Actions"
3. Create a new repository secret named `PYPI_API_TOKEN` with your PyPI token as the value

## Manual triggering

You can also manually trigger the publish workflow from the GitHub Actions tab.

## Release workflow

The release workflow in `.github/workflows/release.yml` automatically creates releases when you push to the main branch.

## Troubleshooting

If the publish workflow fails:

1. Check that your PYPI_API_TOKEN secret is correctly set
2. Verify that the package builds correctly locally
3. Check the workflow logs for specific error messages
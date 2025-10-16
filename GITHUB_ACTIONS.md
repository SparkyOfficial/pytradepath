# GitHub Actions for PyPI Publishing

This repository is configured to automatically publish releases to PyPI using GitHub Actions with OIDC (OpenID Connect) authentication.

## How it works

1. When you create a new release on GitHub, the workflow in `.github/workflows/publish.yml` is triggered
2. The workflow builds the package using the standard Python build tools
3. The package is then uploaded to PyPI using OIDC authentication (no API tokens needed)

## OIDC Authentication

This repository uses OpenID Connect (OIDC) for secure, passwordless publishing to PyPI. This is more secure than using API tokens because:

- No long-lived API tokens to manage or rotate
- Trust is established through configuration, not secrets
- Automatic authentication without manual token management

## Manual triggering

You can also manually trigger the publish workflow from the GitHub Actions tab.

## Release workflow

The release workflow in `.github/workflows/release.yml` automatically creates releases when you push to the main branch.

## Troubleshooting

If the publish workflow fails:

1. Verify that the trusted publisher is correctly configured on PyPI
2. Check that the workflow file path matches exactly
3. Ensure you're using the `pypa/gh-action-pypi-publish` action
4. Check the workflow logs for specific error messages

## Removing API token

Since we're using OIDC, you can remove the `PYPI_API_TOKEN` secret from your repository settings if you previously added it.
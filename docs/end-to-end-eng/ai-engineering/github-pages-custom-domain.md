# Setting Up GitHub Pages with Custom Domain and CI/CD

*Published: November 16, 2025*

In this post, we'll walk through setting up a professional documentation site using MkDocs, GitHub Pages, and a custom domain (aiops-insights.com). This setup includes continuous deployment through GitHub Actions.

## Overview

We'll be working with the following components:
- **MkDocs**: For building static documentation
- **GitHub Pages**: For hosting the site
- **Custom Domain**: aiops-insights.com
- **CI/CD**: GitHub Actions for automated builds and deployments

## Step 1: Initial Setup

### Project Structure
```
aiopsinsights/
├── .github/
│   └── workflows/
│       └── deploy.yml    # GitHub Actions workflow
├── docs/
│   ├── index.md         # Homepage
│   ├── ai-engineering/  # AI Engineering content
│   ├── ai-papers/       # Research papers
│   └── finops/          # FinOps resources
└── mkdocs.yml           # MkDocs configuration
```

### Key Configuration Files

#### `mkdocs.yml`
```yaml
site_name: AIOps Insights
site_url: https://aiops-insights.com
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.indexes
    - search.highlight
    - navigation.footer
    - navigation.tabs.sticky
    - header.autohide
```

## Step 2: Setting Up GitHub Actions

We're using GitHub Actions to automate the build and deployment process. The workflow:

1. Triggers on pushes to `main` branch
2. Sets up Python environment
3. Installs MkDocs and plugins
4. Builds the site
5. Deploys to GitHub Pages

```yaml
# .github/workflows/deploy.yml
name: Deploy MkDocs site to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with: { python-version: '3.10' }
      - run: |
          pip install mkdocs-material mkdocs-git-revision-date-localized-plugin
          mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        with: { path: ./site }

  deploy:
    environment: { name: github-pages, url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build-deploy
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

## Step 3: Configuring Custom Domain

1. **DNS Configuration**
   - Add A records pointing to GitHub Pages IPs:
     ```
     @    A     185.199.108.153
     @    A     185.199.109.153
     @    A     185.199.110.153
     @    A     185.199.111.153
     www  CNAME yourusername.github.io
     ```

2. **GitHub Pages Settings**
   - Go to Repository Settings > Pages
   - Set source to 'GitHub Actions'
   - Add custom domain: `aiops-insights.com`
   - Enforce HTTPS (recommended)

## Common Issues & Solutions

### 1. 404 Errors
- Verify the `gh-pages` branch exists
- Check GitHub Pages build logs for errors
- Ensure `CNAME` file is in the root of the built site

### 2. Build Failures
- Check Python version compatibility
- Verify all required Python packages are in `requirements.txt`
- Look for MkDocs build errors in the Actions logs

### 3. Custom Domain Not Working
- Verify DNS propagation with `dig`
- Check GitHub Pages settings for domain verification
- Ensure the domain is properly configured in the repository settings

## Next Steps

1. **Content Organization**
   - Add more documentation sections
   - Implement search functionality
   - Add analytics

2. **Enhancements**
   - Add dark/light mode toggle
   - Implement versioning for documentation
   - Add a blog section

## Conclusion

This setup provides a robust foundation for hosting documentation with automated deployments. The combination of MkDocs, GitHub Pages, and GitHub Actions creates a seamless workflow for maintaining up-to-date documentation.

For more details, check out the [GitHub repository](https://github.com/himy2u/aiopsinsights).

---
*Himanshu is a Data Leader with expertise in AI/ML, data engineering, and cloud infrastructure. Connect with me on [Twitter](https://x.com/himanshuptech) or [LinkedIn](https://www.linkedin.com/in/hrnp).*

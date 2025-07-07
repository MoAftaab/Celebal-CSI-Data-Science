# Deployment Guide for Breast Cancer Classifier App

This guide provides step-by-step instructions for deploying the Breast Cancer Classifier Streamlit app to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. The project code pushed to a GitHub repository
3. A Streamlit Cloud account (free tier available)

## Deployment Steps

### 1. Push the Code to GitHub

1. Create a new GitHub repository (if you don't have one already)
2. Initialize Git in your project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

### 2. Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account
2. Click on "New app" button
3. Select the repository, branch, and file path:
   - Repository: Choose your GitHub repository
   - Branch: `main` (or your default branch)
   - Main file path: `week_7/app.py`
4. Advanced settings (optional):
   - You can customize the app name
   - Set environment variables if needed
5. Click "Deploy"

Your app will be deployed in a few minutes, and you'll receive a URL to access it.

### 3. Handling Project Structure

Since the app relies on model files from the `week 6` directory, you need to ensure these files are included in your GitHub repository:

- `week 6/models/best_model.joblib`
- `week 6/models/scaler.joblib`

If you encounter any issues with importing modules from different directories, you may need to adjust the import paths in the code.

### 4. Updating the Deployed App

When you make changes to your code:

1. Commit and push changes to GitHub:
   ```bash
   git add .
   git commit -m "Update app with new features"
   git push
   ```
2. Streamlit Cloud will automatically detect changes and rebuild your app

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all required packages are listed in `requirements.txt`
2. **File Path Issues**: Check that relative file paths are correct
3. **Import Errors**: Verify that the import statements are correctly handling the project structure
4. **Memory Limits**: If your app exceeds memory limits, optimize your code or upgrade your Streamlit Cloud plan

### Viewing Logs

Streamlit Cloud provides logs for debugging:

1. Go to your app on Streamlit Cloud
2. Click on "Manage app" in the top-right corner
3. Select the "Logs" tab to view runtime logs

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community](https://discuss.streamlit.io/) for support 
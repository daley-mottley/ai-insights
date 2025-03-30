<h1 align="center">
AI Insights <br> Report Generator <br> 🥸</span>
</h1>

<div align="center">
<p align="center">
<a href="https://github.com/daley-mottley/ai-insights/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeature+Request%5D+">Request Feature</a>
     ·
    <a href="https://ai-insights-production.up.railway.app/" target="blank">View Demo</a>
    ·
    <a href="https://github.com/daley-mottley/ai-insights/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml&title=%5BBug%5D+">Report Bug</a>
 
</p>

   <img src="https://res.cloudinary.com/dzpafdvkm/image/upload/v1727899164/Portfolio/ai-insights-screenshot.png" alt="AI Insights Screenshot" />
</div>

## Project Overview 🧐
AI Insights is a web-based application designed to provide valuable insights for your business using AI-driven analytics. The application leverages multiple data sources to deliver comprehensive reports and visualizations, helping users make data-driven decisions with ease. By default, it uses Google’s Gemini model (`gemini-1.5-pro`) for AI capabilities, with OpenAI as an optional alternative.

## Setup Instructions 📄

### Prerequisites 
- Python 3.x
- Poetry package manager (recommended) or pip
- Google Sheets API credentials (either through environment variables or a `credentials.json` file)
- Google Gemini API Key (required for default functionality)
- OpenAI API Key (optional, if using OpenAI instead of Gemini)

### Installation

1. **Clone the repository:**
   ```bash
   # Clones the ai-insights repository from GitHub to your local machine
   git clone https://github.com/daley-mottley/ai-insights.git
   # Changes the current directory to the cloned repository
   cd ai-insights
   ```

2. **Install dependencies:**
   ```bash
   # Installs the project dependencies listed in requirements.txt using pip
   # Note: Poetry is preferred (run `poetry install` instead) as it aligns with pyproject.toml
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Configure the necessary environment variables. You can set them in your environment or create a `.env` file in the root directory. Required variables include:

   - `GOOGLE_SHEETS_TYPE`
   - `GOOGLE_SHEETS_PROJECT_ID`
   - `GOOGLE_SHEETS_PRIVATE_KEY_ID`
   - `GOOGLE_SHEETS_PRIVATE_KEY`
   - `GOOGLE_SHEETS_CLIENT_EMAIL`
   - `GOOGLE_SHEETS_CLIENT_ID`
   - `GOOGLE_SHEETS_AUTH_URI`
   - `GOOGLE_SHEETS_TOKEN_URI`
   - `GOOGLE_SHEETS_AUTH_PROVIDER_X509_CERT_URL`
   - `GOOGLE_SHEETS_CLIENT_X509_CERT_URL`
   - `GEMINI_API_KEY` (required for default Gemini model)
   - `OPENAI_API_KEY` (optional, required if using OpenAI)
   - `LLM_MODEL` (optional, set to `'openai'` to use OpenAI; defaults to `'gemini'`)
   - `PROTONMAIL_ADDRESS`
   - `PROTONMAIL_PASSWORD`
   - `MONGODB_URI`

   Alternatively, you can store Google Sheets credentials in a `credentials.json` file in the root directory.

4. **Run the application:**
   ```bash
   # Launches the Flask application using Python, defaulting to port 5000
   python app.py
   ```

5. **Access the application:**
   Open your web browser and go to `http://localhost:5000` to view the application.

**Note:** The application uses Google’s Gemini model (`gemini-1.5-pro`) by default for AI-driven report generation. To switch to OpenAI (e.g., `gpt-3.5-turbo`), set `LLM_MODEL=openai` and provide an `OPENAI_API_KEY` in your environment variables.

## Contributions 🧑‍🔧👷‍♀️🏗️🏢

Contributions are welcome! It only takes five (5) steps!

To contribute:

1) Fork the repository.

2) Create a new branch: `git checkout -b my-feature-branch`.

3) Make your changes and commit them: `git commit -m 'Add some feature'`.

4) Push to the branch: `git push origin my-feature-branch`.

5) Open a pull request.

<p align="center" ><strong><em>Please read our <a href="https://github.com/daley-mottley/ai-insights/blob/main/CONTRIBUTION.md" >Contributing Guidelines</a> to get started!</em></strong> 🚀</p>

<p align="center">🫶 <em>Thank you for your support! </em>🙌 </p>
<hr>
<h2 align="center"> 🌎 Let's Stay Connected 🫸🫷 </h2>

<p align="center"> If you like this project and would like to see more features or show your support.</p>
<p align="center"> Feel free to reach out to the developer(s) and give this project a ⭐!</p>

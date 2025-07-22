
# My EBITA

An AI Financial Sidekick. My EBITA stands for Earnings Beat Indicator & Text Analyzer.

This application is designed for the discerning investor who wants to look past the numbers. It provides a simple, clean interface to acquire earnings call transcripts and run a dual analysis using both Gemini and ChatGPT. The goal is to provide a comprehensive, "between-the-lines" look at management sentiment, confidence, and potential red flags, all on a single, easy-to-read report.

TODO screenshot

## Installation

To install this app, simply clone the repository and install the dependencies in requirements.txt using `pip`

```bash
   pip install -r requirements.txt
```
You'll also need to configure your API keys for the AI models and the data acquisition service in a .env file.

## Usage

_The app is currently a local Flask application, not publicly deployed._
TODO URL if deployed

To use this app, run the following command `python app.py`
>- The app supports multiple users with individual analysis reports and collections. You will be prompted to log in or register a new account.
>- The core of the app is the Dashboard, where you can acquire new earnings call transcripts and run your analysis.

Acquiring a Transcript
>- On the dashboard, enter a Ticker Symbol, Fiscal Year, and Fiscal Quarter. The app will fetch the transcript from the API Ninjas service.
>- If a transcript already exists, you will be notified and can proceed to analysis.
>- Transcripts are stored in your private collection, and can be viewed or deleted from the dashboard.

Running an Analysis
>- Once a transcript is acquired, enter its Transcript ID into the "Analyze Transcript" form.
>- You can optionally provide a custom analysis prompt. If left blank, a default prompt will be used to generate a structured report.
>- The app will then send the transcript to both Gemini and ChatGPT for a dual AI analysis. The result is a single, comparative report.
>- Analysis reports are stored per user and can be viewed or deleted from the dashboard.

## Project Status

As of _23-JUL-2025_, project is: _MVP_

## Room for Improvement

>- Integration with more robust data APIs for company profiles (industry, sector, etc.).
>- Implement a user-driven search, sort, and filter functionality for transcripts and reports.
>- Add a user profile management page with a password reset function.
>- Develop a custom, conversational analysis feature where users can ask follow-up questions to the AI models.
>- Add a feature to download reports.
>- Create a feature to flag transcripts for future analysis or review.

> - Log and API limits
> - Apply CSS framework (Bootstrap or Bulma)
> - Include images from Unsplash
> - Include icons from Font Awesome
> - Apply coloring from Adobe Color
> - API Docs
> - API for industry, sector
> - User questionnaire

## Acknowledgements

A special thanks to the entire team at Masterschool, and especially to my AI Mentor Zisis Batzos, for providing the guidance (and the patience) in building this app.

## Contributing

I welcome any contributions! If you'd like to contribute to this project, please reach out to [email@edgroell.com](mailto:email@edgroell.com)
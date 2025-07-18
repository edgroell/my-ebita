"""
Project: My EBITA app - Earnings Beat Indicator & Text Analyzer
App: Your AI Financial Sidekick
by Ed Groell
Latest: 18-JUL-2025
"""


from flask import Flask, render_template, Response

app = Flask(__name__)


# -----------------------------------------------------
# Status
# -----------------------------------------------------

@app.route('/api/v1/status')
def status():
    return {"status": "ok", "message": "EBITA API is operational, barely."}, 200

# -----------------------------------------------------
# Homepage
# -----------------------------------------------------

@app.route('/')
def index():
    # render-template index.html
    pass

# -----------------------------------------------------
# Authentication & User Management
# -----------------------------------------------------

@app.route('/api/v1/auth/register', methods=['POST'])

@app.route('/api/v1/auth/login', methods=['POST'])

@app.route('/api/v1/auth/logout', methods=['POST'])

@app.route('/api/v1/auth/users/me', methods=['GET'])

# -----------------------------------------------------
# Companies
# -----------------------------------------------------

@app.route('/api/v1/companies', methods=['GET'])

@app.route('/api/v1/companies/{ticker_symbol}', methods=['GET'])

# -----------------------------------------------------
# Earnings Call Transcripts
# -----------------------------------------------------

@app.route('/api/v1/companies/{ticker_symbol}/transcripts', methods=['GET'])

@app.route('/api/v1/transcripts/{transcript_id}', methods=['GET'])

# -----------------------------------------------------
# Analysis Reports
# -----------------------------------------------------

@app.route('/api/v1/analysis', methods=['GET', 'POST'])

@app.route('/api/v1/analysis/{report_id}', methods=['GET'])

# -----------------------------------------------------
# User Watchlists
# -----------------------------------------------------

@app.route('/api/v1/watchlists', methods=['GET', 'POST'])

@app.route('/api/v1/watchlists/{watchlist_id}', methods=['PUT', 'DELETE'])

@app.route('/api/v1/watchlists/{watchlist_id}/stocks', methods=['POST', 'DELETE'])

# -----------------------------------------------------
# Error Handlers
# -----------------------------------------------------

@app.errorhandler(404)
def page_not_found(e) -> Response:
    """
    Handles 404 errors (Page Not Found).
    Renders a custom 404 page and returns the 404 status code.
    The 'e' parameter is the error object, which is required by Flask.
    """
    print(f"An error occurred: {e}")

    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e) -> Response:
    """
    Handles 500 errors (Internal Server Error).
    This is triggered by unhandled exceptions in your code.
    Renders a custom 500 page and returns the 500 status code.
    """
    print(f"An internal server error occurred: {e}")

    return render_template('500.html'), 500


if __name__ == '__main__':
    with app.app_context():
        data.data_models.db.create_all()

    app.run(debug=True)

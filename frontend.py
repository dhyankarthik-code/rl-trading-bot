from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Backend API base URL (assuming local)
API_BASE = "http://localhost:8000"

@app.route('/')
def hub():
    """
    Homepage dashboard hub.
    """
    return render_template('index.html')

@app.route('/supercharts')
def supercharts():
    """
    Main charting workspace.
    """
    return render_template('supercharts.html')  # Placeholder, create later

@app.route('/screener')
def screener():
    """
    Stock/crypto screener.
    """
    return render_template('screener.html')  # Placeholder

@app.route('/calendar')
def calendar():
    """
    Economic calendar.
    """
    response = requests.get(f"{API_BASE}/calendar")
    events = response.json().get('events', [])
    return render_template('calendar.html', events=events)

@app.route('/news')
def news():
    """
    News feed.
    """
    query = request.args.get('q', 'trading')
    response = requests.get(f"{API_BASE}/news", params={'query': query})
    articles = response.json().get('articles', [])
    return render_template('news.html', articles=articles)

@app.route('/portfolio')
def portfolio():
    """
    Portfolio management.
    """
    return render_template('portfolio.html')  # Placeholder

@app.route('/options')
def options():
    """
    Options builder.
    """
    return render_template('options.html')  # Placeholder

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
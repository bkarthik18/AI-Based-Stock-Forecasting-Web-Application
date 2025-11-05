from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import uuid
import numpy as np
import secrets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime as dt
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import traceback
import requests
import pandas as pd

# -------------------- Flask / DB setup --------------------
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydb.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Prefer env var if present; fallback to your provided key
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY") or "6f7fe45f89a2df93804656a91425c3357613cd7f"

# -------------------- Models --------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

with app.app_context():
    db.create_all()

# -------------------- Error handlers --------------------
@app.errorhandler(404)
def handle_404(e):
    return jsonify({"success": False, "message": "Not found"}), 404

@app.errorhandler(405)
def handle_405(e):
    return jsonify({"success": False, "message": "Method not allowed"}), 405

@app.errorhandler(500)
def handle_500(e):
    return jsonify({"success": False, "message": "Internal server error"}), 500

# -------------------- Auth helper --------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        token = None
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                token = auth_header
        if not token or len(token) != 64:
            return jsonify({'success': False, 'message': 'Invalid or missing token'}), 401
        return f(*args, **kwargs)
    return decorated

# -------------------- Auth routes --------------------
@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json(silent=True) or {}
        name = data.get('name', '').strip()
        phone = data.get('phone', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()

        if not name or not phone or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 409

        password_hash = generate_password_hash(password)
        user = User(name=name, phone=phone, email=email, password_hash=password_hash)
        db.session.add(user)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Signup successful!'}), 201
    except Exception as e:
        print("❌ Signup Error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error occurred'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json(silent=True) or {}
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            token = secrets.token_hex(32)
            return jsonify({'success': True, 'name': user.name, 'token': token})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    except Exception as e:
        print("❌ Login Error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error occurred'}), 500

# -------------------- ML helpers --------------------
def create_dataset(dataset: np.ndarray, time_step: int = 60):
    """
    dataset: shape (n, 1), already scaled
    returns: X shape (n - time_step, time_step, 1), y shape (n - time_step, 1)
    """
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step, 0])
        y.append(dataset[i + time_step, 0])
    X = np.array(X)
    y = np.array(y)
    if X.size == 0 or y.size == 0:
        raise ValueError("Not enough data to create training sequences. Try a smaller time_step or provide more history.")
    return X, y

def fetch_tiingo_close_df(symbol: str, start: dt.datetime, end: dt.datetime, api_key: str) -> pd.DataFrame:
    """
    Fetch OHLCV from Tiingo and return a DataFrame with a single 'close' column indexed by date.
    """
    if not api_key:
        raise ValueError("Missing Tiingo API key.")

    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {
        "token": api_key,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d")
    }
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code == 404:
        raise ValueError(f"No data found for symbol '{symbol}'.")
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"No data returned for symbol '{symbol}' in the given date range.")

    df = pd.DataFrame(data)
    # Tiingo returns 'date' ISO string; ensure datetime and sort
    if 'date' not in df.columns:
        raise ValueError("Response missing 'date' field.")
    if 'close' not in df.columns:
        # Fallbacks (rare)
        for k in ['adjClose', 'Adj Close', 'adj_close']:
            if k in df.columns:
                df.rename(columns={k: 'close'}, inplace=True)
                break
        else:
            raise ValueError(f"Response missing 'close' field. Columns: {list(df.columns)}")

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # Keep only close, drop NaNs, forward-fill if needed
    close_df = df[['close']].copy()
    close_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    close_df.dropna(inplace=True)
    if close_df.empty:
        raise ValueError("Close price series is empty after cleaning.")
    return close_df

# -------------------- Prediction route --------------------
@app.route('/predict', methods=['POST'])
@token_required
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        company = payload.get('company', '').strip().upper()
        if not company:
            return jsonify({'error': 'Company symbol is required'}), 400

        end = dt.datetime.now()
        start = end - dt.timedelta(days=1000)

        # 1) Fetch data robustly (no pandas_datareader)
        data_close = fetch_tiingo_close_df(company, start, end, TIINGO_API_KEY)

        # 2) Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_close.values)  # shape (n,1)

        # Ensure we have enough data; shrink time_step if needed
        time_step = 60
        if len(scaled_data) <= time_step:
            time_step = max(5, len(scaled_data) - 1)

        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # 3) Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=5, batch_size=64, verbose=0)

        # 4) Forecast 30 steps ahead — using np.concatenate
        temp_input = scaled_data[-time_step:].reshape(-1, 1)  # shape (time_step, 1)

        print("DEBUG: type of np.concatenate =", type(np.concatenate))
        print("DEBUG: np.concatenate repr   =", np.concatenate)
        
        forecast_scaled = []

        for _ in range(30):
            input_array = temp_input[-time_step:].reshape(1, time_step, 1)
            predicted = model.predict(input_array, verbose=0)  # shape (1,1)

            # concatenate instead of vstack
            temp_input = np.concatenate([temp_input, predicted], axis=0)
            forecast_scaled.append(predicted[0, 0])

        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
        forecast = scaler.inverse_transform(forecast_scaled).flatten().tolist()

        # 5) Plot
        os.makedirs('static', exist_ok=True)
        plot_filename = f"forecast_plot_{uuid.uuid4().hex}.png"
        plot_path = os.path.join('static', plot_filename)

        plt.figure(figsize=(10, 4))
        plt.plot(forecast, label=f'{company} Forecast')
        plt.title(f'{company} - 30 Day Forecast')
        plt.xlabel('Day')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return jsonify({
            'forecast': [round(float(price), 2) for price in forecast],
            'forecast_plot': plot_path
        })

    except requests.HTTPError as http_err:
        print("❌ HTTP Error:", http_err)
        traceback.print_exc()
        return jsonify({'error': f'HTTP error from Tiingo: {http_err}'}), 502
    except Exception as e:
        print("❌ Prediction Error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# -------------------- Entrypoint --------------------
if __name__ == '__main__':
    # Bind to 0.0.0.0 if you plan to hit it from a different device/network
    app.run(host='0.0.0.0', port=5001)

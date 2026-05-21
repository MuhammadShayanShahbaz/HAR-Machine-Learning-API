# 🏃‍♂️ Real-Time Human Activity Recognition (HAR) Architecture

A full-stack, edge-to-cloud IoT architecture that predicts human movement in real-time. This system reads physical smartphone gyroscope and accelerometer sensors, extracts statistical features locally in the browser via edge computing, and streams the data through a persistent WebSocket tunnel to a machine learning AI in the cloud.

## 🧠 System Architecture

This project utilizes a **3-Tier Separation of Concerns** model to ensure low latency and scalable inference without overloading the server.

1. **The Edge Client (Frontend):** A zero-dependency HTML/JS client hosted on GitHub Pages. It acts as an edge-computing node, buffering physical sensor data at 60Hz, extracting 34 statistical features (Means, Standard Deviations, Jerk, Magnitude), and normalizing the data to a `[-1, 1]` scale before transmission.
2. **The Inference Engine (Backend):** A containerized Python FastAPI server. It maintains open WebSocket connections, parses incoming feature arrays, and feeds them into a Scikit-Learn Random Forest model trained on the UCI HAR Dataset. 
3. **The Data Access Layer (Database):** A PostgreSQL database managed via SQLAlchemy. To prevent database flooding from high-frequency IoT data, it utilizes **State-Change Logging**, only executing write operations when the user's physical activity state actively changes (e.g., transitioning from `SITTING` to `WALKING`).

## 🛠️ Tech Stack

* **Machine Learning:** Python, Scikit-Learn, NumPy, Pandas
* **Backend API:** FastAPI, Uvicorn, WebSockets
* **Database:** PostgreSQL, SQLAlchemy (ORM)
* **Frontend:** HTML5 DeviceMotion API, Vanilla JavaScript, TailwindCSS
* **Deployment:** Docker, Render (Backend), GitHub Pages (Frontend)

## ✨ Core Features

* **Real-Time Telemetry:** Bypasses standard HTTP REST overhead by utilizing a bidirectional WebSocket `wss://` tunnel for instant inference.
* **Edge Computing:** Saves massive amounts of server CPU and bandwidth by calculating the 34 time-domain and frequency-domain variables directly on the user's smartphone processor.
* **Gait Cycle Memory:** Implements a 128-frame (~2.5 second) rolling queue memory buffer to accurately capture the rhythmic sine-wave of human walking cycles.
* **Gravity Compensation:** Mathematically accounts for gravitational leakage and dataset normalization offsets to accurately predict static states (`STANDING`, `LAYING`).
* **Auto-Healing Connections:** The frontend features exponential backoff algorithms to seamlessly reconnect if the user's mobile network drops.

## 🚀 Run It Locally

Because the backend is fully containerized, you can spin up the entire API environment on any machine using Docker.

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/har-machine-learning-api.git](https://github.com/yourusername/har-machine-learning-api.git)
cd har-machine-learning-api

```
## 2. Set up your Environment Variables
* ** Create a .env file in the root directory and add your credentials:
Plaintext
DATABASE_URL=postgresql://user:password@localhost:5432/hardb
API_KEY=your-secure-secret-key

## 3. Build and Run the Docker Container

Bash
docker build -t har-api .
docker run -p 8000:8000 --env-file .env har-api

## 4. Connect the Client
* **Open index.html in your mobile device's web browser, grant sensor permissions, and place the device in your pocket to begin real-time inference streaming.

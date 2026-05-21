# 1. Start with a lightweight Linux OS that has Python 3.9 pre-installed
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (This caches your installations to save build time later)
COPY requirements.txt .

# 4. Install all the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Open port 8000 so the outside world can talk to the container
EXPOSE 8000

# 7. The command to boot up your FastAPI server when the container turns on
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
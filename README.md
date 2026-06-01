# Airman — Air Quality ETL Pipeline

Airman is a data engineering and machine learning project for building an air quality data pipeline using local CSV files, Pandas, Docker, and ClickHouse.

The project currently focuses on creating a clean local ETL pipeline that reads raw Beijing air quality data, transforms it into a clean time-series dataset, and loads it into a local ClickHouse database.

---

## Project Goal

The full goal of the Airman project is to build an end-to-end air quality pipeline:

```text
Raw CSV data
→ Data ingestion
→ ETL cleaning pipeline
→ Processed dataset
→ ClickHouse database
→ Forecasting model
→ Dashboard
→ Cloud deployment
```

The project is being built locally first. Cloud storage such as AWS S3 will be added later after the local pipeline is stable.

---

## Current Project Status

### Sprint 1 — Local Data Ingestion

Sprint 1 focused on setting up the project structure and preparing local raw data storage.

Completed:

* Created project folder structure
* Added `data/raw/` for raw CSV files
* Added `data/processed/` for cleaned CSV files
* Added `.gitignore`
* Added `.env.example`
* Added local storage configuration
* Added raw data validation script
* Updated project documentation

Current Sprint 1 flow:

```text
Local raw CSV files
→ data/raw/
→ raw data validation
→ ready for ETL pipeline
```

---

### Sprint 2 — Local ETL Pipeline + ClickHouse

Sprint 2 focuses on building the local ETL pipeline and loading cleaned data into ClickHouse.

Completed:

* Built local ETL pipeline with Pandas
* Loaded raw CSV from `data/raw/`
* Cleaned column names
* Created a proper `datetime` column
* Handled missing values
* Saved cleaned data to `data/processed/air_quality_cleaned.csv`
* Set up ClickHouse using Docker
* Created the `airman.air_quality` table
* Loaded cleaned data into ClickHouse
* Verified table schema and sample query results

Current Sprint 2 flow:

```text
data/raw/*.csv
→ src/pipeline.py
→ data/processed/air_quality_cleaned.csv
→ src/load_clickhouse.py
→ ClickHouse database
→ airman.air_quality table
```

---

## Project Structure

```text
Airman/
├── config/
├── dashboards/
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── docker/
├── jenkins/
├── models/
│   └── .gitkeep
├── notebooks/
│   └── download_data.ipynb
├── src/
│   ├── __init__.py
│   ├── check_raw_data.py
│   ├── ingest.py
│   ├── load.py
│   ├── load_clickhouse.py
│   ├── pipeline.py
│   ├── storage.py
│   └── transform.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

## Tech Stack

Current tools:

* Python
* Pandas
* python-dotenv
* Docker
* ClickHouse
* clickhouse-connect

Planned tools:

* PyTorch
* LSTM forecasting model
* Streamlit dashboard
* AWS S3 or S3-compatible storage
* Jenkins or CI/CD pipeline
* Cloud deployment

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:Fatin-khan/Airman-Air-Quality-Pipeline.git
cd Airman-Air-Quality-Pipeline
```

---

### 2. Create and Activate Virtual Environment

On Windows Git Bash:

```bash
python -m venv venv
source venv/Scripts/activate
```

On Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Create Environment File

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Example `.env`:

```env
PROJECT_NAME=Airman
STORAGE_BACKEND=local
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
```

The `.env` file is private and should not be pushed to GitHub.

---

## Data Storage

Raw and processed datasets are kept locally.

Raw data should be placed inside:

```text
data/raw/
```

Processed data is saved inside:

```text
data/processed/
```

Example processed output:

```text
data/processed/air_quality_cleaned.csv
```

Dataset files are ignored by Git because they can be large and should not be pushed to GitHub.

---

## Run Raw Data Check

To check whether raw CSV files are available and readable:

```bash
python -m src.check_raw_data
```

This script checks:

* CSV files in `data/raw/`
* Number of rows
* Number of columns
* Column names
* Missing values
* First few rows

---

## Run the ETL Pipeline

To clean the raw air quality dataset:

```bash
python -m src.pipeline
```

The ETL pipeline performs:

* Reads raw CSV file from `data/raw/`
* Cleans column names
* Creates a correct `datetime` column
* Sorts data by datetime
* Handles missing values
* Removes unnecessary columns
* Saves cleaned CSV to `data/processed/air_quality_cleaned.csv`

Expected output:

```text
Starting Sprint 2 local ETL pipeline...
Reading raw CSV...
Cleaning column names...
Creating datetime column...
Handling missing values...
Saving cleaned data...
ETL pipeline completed successfully.
```

---

## ClickHouse Setup

ClickHouse runs locally using Docker.

### Start ClickHouse

```bash
docker compose up -d
```

### Check Running Containers

```bash
docker ps
```

You should see:

```text
airman-clickhouse
```

### Test ClickHouse

```bash
docker exec -it airman-clickhouse clickhouse-client --user airman_user --password airman_password --query "SELECT version();"
```

---

## Load Cleaned Data into ClickHouse

After running the ETL pipeline, load the cleaned CSV into ClickHouse:

```bash
python -m src.load_clickhouse
```

Expected result:

```text
Data loaded successfully. Row count in ClickHouse: 473352
```

The cleaned data is loaded into:

```text
Database: airman
Table: air_quality
```

Full table name:

```text
airman.air_quality
```

---

## Verify ClickHouse Data

### Check Row Count

```bash
docker exec -it airman-clickhouse clickhouse-client --user airman_user --password airman_password --query "SELECT count() FROM airman.air_quality;"
```

Expected result:

```text
473352
```

### Check Datetime Range

```bash
docker exec -it airman-clickhouse clickhouse-client --user airman_user --password airman_password --query "SELECT min(datetime), max(datetime) FROM airman.air_quality;"
```

Expected result:

```text
2010-01-01 00:00:00    2017-02-28 23:00:00
```

### Preview Sample Rows

```bash
docker exec -it airman-clickhouse clickhouse-client --user airman_user --password airman_password --query "SELECT city, datetime, pm2_5, temperature_celsius, humidity FROM airman.air_quality LIMIT 5;"
```

---

## Current ClickHouse Schema

The `airman.air_quality` table includes columns such as:

```text
city
country
date
season
pm2_5_concentration_ug_m_3
pm2_5
pm10_concentration_ug_m_3
so2_concentration_ug_m_3
no2_concentration_ug_m_3
co_concentration_ug_m_3
o3_concentration_ug_m_3
humidity
pressure_hpa
wind_direction
wind_speed_m_s
dew_point_celsius
temperature_celsius
datetime
```

The table uses the ClickHouse `MergeTree` engine and is ordered by:

```text
datetime
```

---

## Git Ignore Rules

The project ignores files that should not be pushed to GitHub, including:

```text
.env
venv/
__pycache__/
*.pyc
data/raw/*
data/processed/*
models/*
```

The project keeps `.gitkeep` files so empty folders can still appear on GitHub.

---

## Current Local Pipeline

```text
Raw Beijing air quality CSV
→ data/raw/
→ src/pipeline.py
→ cleaned CSV
→ data/processed/air_quality_cleaned.csv
→ src/load_clickhouse.py
→ ClickHouse
→ airman.air_quality
```

---

## Future Work

Planned next steps:

* Add stronger data quality checks
* Add duplicate timestamp and station checks
* Prepare data for time-series forecasting
* Build PyTorch LSTM forecasting model
* Store model outputs
* Build Streamlit dashboard
* Add AWS S3 or S3-compatible object storage
* Deploy dashboard and services to the cloud

---

## Sprint 3 Preview

Sprint 3 will focus on machine learning preparation and forecasting.

Planned Sprint 3 flow:

```text
ClickHouse / processed CSV
→ feature selection
→ train/test split
→ PyTorch LSTM model
→ PM2.5 forecasting
→ saved model output
```

---

## Notes

This project is currently built using local storage first. AWS S3 integration is planned for a later sprint after the local ETL and database workflow are stable.

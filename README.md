🚀 NewsClassifier-MLOps

An end-to-end MLOps pipeline for text classification with full lifecycle:
data ingestion → preprocessing → model training → evaluation → deployment → monitoring.

<p align="center"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="50" /> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikit-learn/scikit-learn-original.svg" width="50" /> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original.svg" width="50" /> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/kubernetes/kubernetes-plain.svg" width="50" /> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/fastapi/fastapi-original.svg" width="50" /> <img src="https://mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" width="100" /> <img src="https://prefect.io/images/prefect-logo-dark.svg" width="120" /> <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Prometheus_software_logo.svg" width="50" /> <img src="https://grafana.com/static/assets/img/fav32.png" width="40" /> </p>
🔑 Features

Data Science: TF-IDF + scikit-learn classifiers (Logistic, SVM, RandomForest) with GridSearchCV.

Experiment Tracking: MLflow for metrics, parameters, artifacts, and model registry.

Pipeline Orchestration: Prefect flow connecting preprocessing → training → evaluation → deployment.

Model Serving: FastAPI service exposing /predict, /info, /metrics.

Monitoring: Prometheus + Grafana dashboards for API and model health.

DevOps: Docker, Kubernetes manifests, GitHub Actions for CI/CD.

Testing: Pytest for unit tests + Locust for stress/load testing.

⚡ Tech Stack

Python, scikit-learn, pandas, numpy, matplotlib,
MLflow, Prefect, FastAPI, Prometheus, Grafana,
Docker, Kubernetes, GitHub Actions, Pytest, Locust

📌 Status

🔄 Ongoing Project – currently experimenting with datasets and extending evaluation/monitoring.
# Bridgewater-Style AI Lab: System Design, Architecture & Trade-offs

This README provides a high-level overview of a production-lean architecture for an AI-driven macro trading system. It summarizes the design choices, AWS stack, and key trade-offs outlined in the design document.

---

## 1. Overall System Architecture

**Data-to-decision pipeline (daily batch with on-demand inference):**

1. **Ingest** → S3 Bronze (CSV/Parquet) via EventBridge + Lambda.
2. **Transform/Join** → EMR Serverless (Spark) writes Silver/Gold Parquet to S3.
3. **Causal Signals** → Ray tasks (X-Learner/CATE) persisted alongside Gold.
4. **Train & Backtest** → Ray on EKS (PyTorch, Ray Train/Tune; walk-forward validation).
5. **Validate** → Statistical gates (Sharpe, max DD, turnover) gate promotion.
6. **Register** → S3 artifacts + DynamoDB registry (version, metrics, lineage).
7. **Deploy** → ECS Fargate (FastAPI) blue/green; optional Ray Serve on EKS.
8. **Observe** → CloudWatch logs/metrics, Evidently drift reports, alarms.

---

## 2. Training Loop Design

* Load Gold features + CATE from S3 (point-in-time joins only).
* Initialize PyTorch model (MLP/Transformer-lite); deterministic seeds.
* Distributed train with Ray Train; profile with torch.profiler.
* Export to TorchScript + ONNX; package inference graph.
* Ray Tune HPO on limited grid; select by walk-forward Sharpe subject to DD/turnover constraints.
* Persist artifacts; write registry entry (metrics, data slice, git SHA).
* Emit Step Functions event for deploy.

---

## 3. Theoretical Underpinnings

* **Macro time-series** (inflation, growth, rates, liquidity) → regime features.
* **Causal inference (ATE/CATE)** quantifies conditional effects of shocks (e.g., CPI surprise) on asset returns, improving explanatory power vs. pure correlation.
* **Deep learning** maps macro + causal features → asset tilts; risk constraints prevent overfit.
* **Market PnL** closes the loop; periodic retraining adapts to regime drift; validation gates protect production.

---

## 4. AWS Components (Chosen Stack)

* **S3**: Data lake (Bronze/Silver/Gold) + model artifacts.
* **EventBridge + Lambda**: Scheduled ingest jobs.
* **EMR Serverless (Spark)**: ETL, joins, window features → Silver/Gold.
* **EKS + Karpenter (Ray)**: Distributed training/backtests/HPO; optional Ray Serve.
* **ECS Fargate (FastAPI)**: Inference API with blue/green via ALB target groups.
* **Step Functions**: End-to-end orchestration and validation gates.
* **DynamoDB**: Lightweight model registry (version, metrics, lineage).
* **ECR**: Docker images for training/inference jobs.
* **Secrets Manager**: API keys/creds; least-privilege IAM roles.
* **CloudWatch**: Logs, metrics, alarms; Evidently for drift reports.

---

## 5. Design Alternatives: Pros & Cons

### Inference Serving

* **Chosen: FastAPI on ECS Fargate** (control, portability, blue/green). Alternatives: SageMaker, Ray Serve, Lambda.

### ETL/Feature Engineering

* **Chosen: EMR Serverless (Spark)** (pay-per-job, scalable). Alternatives: Databricks, AWS Batch.

### Training/HPO

* **Chosen: Ray on EKS** (flexibility, Python-native). Alternatives: SageMaker Training, AWS Batch.

### Model Registry

* **Chosen: DynamoDB + S3** (lightweight, custom schema). Alternatives: MLflow, SageMaker Model Registry.

### Orchestration

* **Chosen: Step Functions** (serverless, simple retries). Alternatives: MWAA (Airflow), Argo Workflows.

### Monitoring/Drift

* **Chosen: CloudWatch + Evidently** (native alarms). Alternatives: SageMaker Model Monitor, Prometheus/Grafana.

### Data Catalog/Governance

* **Chosen: No Glue** (simplify for single team demo). Alternatives: Glue Catalog + Lake Formation.

### Feature Store

* **Chosen: None** (S3 + Parquet sufficient). Alternatives: Feast, SageMaker Feature Store.

### Deployment Strategy

* **Chosen: ALB Blue/Green**. Alternatives: CodeDeploy Canary, SageMaker Endpoint Shadow.

---

## 6. Recommendations & Upgrade Path

* Start with the chosen stack for a **single-team pilot**: EMR Serverless + Ray on EKS; ECS FastAPI serving; Step Functions orchestration; DynamoDB registry; CloudWatch + Evidently monitoring.
* Add **Glue Catalog + Lake Formation** for multi-team scaling.
* Consider **MLflow** if experiment tracking volume grows.
* Evaluate **SageMaker Endpoints/Monitor** if seeking managed serving/monitoring.
* Explore **Ray Serve** when training and serving converge on EKS.

---

## Summary

This architecture balances lean AWS-native components with flexibility for scaling. The stack emphasizes **pay-per-use, minimal ops burden, and clear upgrade paths**, making it well-suited as a production-lean system for causal macro trading research.

export const MACRO_CAUSAL_CONSTANTS = {
  // Environment configurations
  ENVIRONMENTS: {
    DEV: 'dev',
    STAGING: 'staging',
    PROD: 'prod'
  },

  // S3 bucket configurations
  S3: {
    BRONZE_BUCKET_PREFIX: 'bronze',
    SILVER_BUCKET_PREFIX: 'silver',
    GOLD_BUCKET_PREFIX: 'gold',
    ARTIFACTS_BUCKET_PREFIX: 'artifacts',
    LOGS_BUCKET_PREFIX: 'logs',
    DATA_RETENTION_DAYS: 90,
    MODEL_RETENTION_DAYS: 365
  },

  // EMR configurations
  EMR: {
    RELEASE_LABEL: 'emr-6.15.0',
    SPARK_VERSION: '3.4.0',
    DRIVER_CPU: '4vCPU',
    DRIVER_MEMORY: '16GB',
    WORKER_CPU: '4vCPU',
    WORKER_MEMORY: '16GB',
    MAX_CPU: '200vCPU',
    MAX_MEMORY: '800GB'
  },

  // EKS configurations
  EKS: {
    KUBERNETES_VERSION: '1.27',
    INSTANCE_TYPE: 'm5.large',
    MIN_SIZE: 1,
    MAX_SIZE: 10,
    DESIRED_SIZE: 2
  },

  // Ray configurations
  RAY: {
    HEAD_CPU: '4',
    HEAD_MEMORY: '16Gi',
    WORKER_CPU: '4',
    WORKER_MEMORY: '16Gi',
    MIN_WORKERS: 1,
    MAX_WORKERS: 10
  },

  // ECS configurations
  ECS: {
    CPU: 1024,
    MEMORY_MIB: 2048,
    DESIRED_COUNT: 2,
    MAX_CAPACITY: 10
  },

  // DynamoDB configurations
  DYNAMODB: {
    READ_CAPACITY: 5,
    WRITE_CAPACITY: 5,
    TTL_ATTRIBUTE: 'ttl',
    TTL_DAYS: 365
  },

  // Lambda configurations
  LAMBDA: {
    TIMEOUT_MINUTES: 5,
    MEMORY_MB: 1024,
    RUNTIME: 'python3.9'
  },

  // CloudWatch configurations
  CLOUDWATCH: {
    NAMESPACE: 'MacroCausal',
    METRICS_RETENTION_DAYS: 30,
    LOG_RETENTION_DAYS: 30
  },

  // Step Functions configurations
  STEP_FUNCTIONS: {
    EXECUTION_TIMEOUT_MINUTES: 60,
    STATE_MACHINE_TIMEOUT_MINUTES: 120
  }
} as const;

export const RESOURCE_NAMES = {
  // Stack names
  DATA_LAKE_STACK: 'data-lake-stack',
  ML_TRAINING_STACK: 'ml-training-stack',
  INFERENCE_STACK: 'inference-stack',
  MONITORING_STACK: 'monitoring-stack',
  ORCHESTRATION_STACK: 'orchestration-stack',

  // Construct names
  DATA_LAKE_CONSTRUCT: 'data-lake',
  DATA_INGESTION_CONSTRUCT: 'data-ingestion',
  ML_TRAINING_CONSTRUCT: 'ml-training',
  MODEL_SAVING_CONSTRUCT: 'model-saving',
  INFERENCE_CONSTRUCT: 'inference',
  MONITORING_CONSTRUCT: 'monitoring',
  ORCHESTRATION_CONSTRUCT: 'orchestration',

  // Resource names
  BRONZE_BUCKET: 'bronze-bucket',
  SILVER_BUCKET: 'silver-bucket',
  GOLD_BUCKET: 'gold-bucket',
  ARTIFACTS_BUCKET: 'artifacts-bucket',
  LOGS_BUCKET: 'logs-bucket',
  EMR_APPLICATION: 'emr-application',
  EKS_CLUSTER: 'eks-cluster',
  RAY_CLUSTER: 'ray-cluster',
  ECS_CLUSTER: 'ecs-cluster',
  ALB: 'alb',
  REGISTRY_TABLE: 'registry-table',
  ALERT_TOPIC: 'alert-topic',
  STATE_MACHINE: 'state-machine'
} as const;

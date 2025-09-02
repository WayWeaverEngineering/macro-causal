import { Duration } from "aws-cdk-lib"

export class AwsConfig {
  static readonly WEB_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:us-east-1:715067592333:certificate/6ff63db0-df16-4fc5-8302-bd14b3da1664"
  static readonly API_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:us-east-1:715067592333:certificate/fbf202e6-1365-4265-827a-7510cd0f4ff7"
  static readonly INFERENCE_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:us-east-1:715067592333:certificate/0832e05e-e0c0-4298-bf4e-44266f1a93de"

  static readonly OPENAI_API_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:715067592333:secret:macro-ai-analyst-openai-api-secrets-levvJC"
  static readonly OPENAI_API_SECRET_ID = "macro-ai-analyst-openai-api-secrets"

  static readonly FRED_API_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:715067592333:secret:fred-api-secrets-A1g2T4"
  static readonly FRED_API_SECRET_ID = "fred-api-secrets"

  static readonly QUEUE_TIMEOUT_MINS = Duration.minutes(15)
}
import { Duration } from "aws-cdk-lib"

export class AwsConfig {
  static readonly WEB_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:us-east-1:715067592333:certificate/6ff63db0-df16-4fc5-8302-bd14b3da1664"
  static readonly API_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:us-east-1:715067592333:certificate/62d54e12-aa93-4e84-a840-b33fc5833e0e"

  static readonly OPENAI_API_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:715067592333:secret:sec-ai-analyst-openai-api-secrets-sV98gJ"
  static readonly OPENAI_API_SECRET_ID = "sec-ai-analyst-openai-api-secrets"

  static readonly FRED_API_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:715067592333:secret:fred-api-secrets-A1g2T4"
  static readonly FRED_API_SECRET_ID = "fred-api-secrets"

  static readonly QUEUE_TIMEOUT_MINS = Duration.minutes(15)
}
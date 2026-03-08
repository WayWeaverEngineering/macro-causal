import { Duration } from "aws-cdk-lib"

export class AwsConfig {
  static readonly OPENAI_API_SECRET_ARN = "arn:aws:secretsmanager:ap-southeast-1:978212996804:secret:harry-finance-demos-openai-api-secrets-d5TDoD"
  static readonly OPENAI_API_SECRET_ID = "harry-finance-demos-openai-api-secrets"

  static readonly FRED_API_SECRET_ARN = "arn:aws:secretsmanager:ap-southeast-1:978212996804:secret:fred-api-secrets-2YVwI5"
  static readonly FRED_API_SECRET_ID = "fred-api-secrets"

  static readonly QUEUE_TIMEOUT_MINS = Duration.minutes(15)

  static readonly INFERENCE_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:ap-southeast-1:978212996804:certificate/53174e67-aba7-46f6-a7ed-e8646fbdedb3"
}
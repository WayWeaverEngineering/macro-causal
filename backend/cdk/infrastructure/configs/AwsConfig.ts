import { Duration } from "aws-cdk-lib"

export class AwsConfig {
  static readonly WEB_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:us-east-1:715067592333:certificate/80bd8f7a-6648-46bb-9500-e31beb66446d"
  static readonly API_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:ap-southeast-1:715067592333:certificate/3d3789eb-2d8e-4564-9690-68eeed6796e4"
  static readonly INFERENCE_DOMAIN_CERTIFICATE_ARN = "arn:aws:acm:us-east-1:715067592333:certificate/0832e05e-e0c0-4298-bf4e-44266f1a93de"

  static readonly OPENAI_API_SECRET_ARN = "arn:aws:secretsmanager:ap-southeast-1:715067592333:secret:harry-finance-demos-openai-api-secrets-RCf4Az"
  static readonly OPENAI_API_SECRET_ID = "harry-finance-demos-openai-api-secrets"

  static readonly FRED_API_SECRET_ARN = "arn:aws:secretsmanager:ap-southeast-1:715067592333:secret:fred-api-secrets-wke1iV"
  static readonly FRED_API_SECRET_ID = "fred-api-secrets"

  static readonly QUEUE_TIMEOUT_MINS = Duration.minutes(15)
}
import { Stack, StackProps } from "aws-cdk-lib"
import { Construct } from "constructs"
import { Certificate } from "aws-cdk-lib/aws-certificatemanager"
import { DefaultIdBuilder } from "../../utils/Naming"
import { HttpApi, DomainName, CorsHttpMethod, HttpMethod } from "aws-cdk-lib/aws-apigatewayv2"
import { Function as LambdaFunction } from "aws-cdk-lib/aws-lambda";
import { HttpLambdaIntegration } from "aws-cdk-lib/aws-apigatewayv2-integrations"
import { AwsConfig } from "../configs/AwsConfig"

interface BackendApiStackProps extends StackProps {
  analysisStatusLambda: LambdaFunction
  analysisSchedulingLambda: LambdaFunction
}

export class BackendApiStack extends Stack {

  constructor(scope: Construct, id: string, props: BackendApiStackProps) {
    super(scope, id, props)

    const certificateId = DefaultIdBuilder.build('api-domain-certificate')
    const domainCertificate = Certificate.fromCertificateArn(
      this,
      certificateId,
      AwsConfig.API_DOMAIN_CERTIFICATE_ARN
    );

    const apiDomainConstructId = DefaultIdBuilder.build('backend-api-domain')
    const backendApiDomain = new DomainName(this, apiDomainConstructId, {
      domainName: 'macro-ai-analyst-api.wayweaver.com',
      certificate: domainCertificate
    });
    
    const backendApiId = DefaultIdBuilder.build('backend-http-api')
    const backendApi = new HttpApi(this, backendApiId, {
      apiName: backendApiId,
      description: 'REST API to expose Macro Filings Diff backend functionalities',
      corsPreflight: {
        allowOrigins: ["*"],
        allowMethods: [CorsHttpMethod.OPTIONS, CorsHttpMethod.GET, CorsHttpMethod.POST],
        allowHeaders: ["Content-Type", "x-api-key"]
      },
      defaultDomainMapping: {
        domainName: backendApiDomain
      }
    });

    const analysisSchedulingLambdaIntegration = new HttpLambdaIntegration(
      DefaultIdBuilder.build('analysis-scheduling-lambda-integration'),
      props.analysisSchedulingLambda
    );

    backendApi.addRoutes({
      path: "/analysis",
      methods: [ HttpMethod.POST],
      integration: analysisSchedulingLambdaIntegration
    })

    const analysisStatusLambdaIntegration = new HttpLambdaIntegration(
      DefaultIdBuilder.build('analysis-status-lambda-integration'),
      props.analysisStatusLambda
    );
    
    backendApi.addRoutes({
      path: "/analysis/{executionId}",
      methods: [ HttpMethod.GET],
      integration: analysisStatusLambdaIntegration
    })
  }
}
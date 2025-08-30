import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';
import { SubnetType } from 'aws-cdk-lib/aws-ec2';

export interface ApiDataCollectionProps {
  environment: string;
  bronzeBucket: s3.Bucket;
  vpc: ec2.IVpc;
  securityGroup: ec2.ISecurityGroup;
}

export class ApiDataCollectionConstruct extends Construct {
  public readonly fredCollector: lambda.Function;
  public readonly worldBankCollector: lambda.Function;
  public readonly yahooFinanceCollector: lambda.Function;
  public readonly dataCollectionStateMachine: sfn.StateMachine;
  public readonly scheduledCollectionRule: events.Rule;
  public readonly apiSecrets: secretsmanager.Secret;

  constructor(scope: Construct, id: string, props: ApiDataCollectionProps) {
    super(scope, id);

    // Create Secrets Manager secret for API keys
    this.apiSecrets = new secretsmanager.Secret(this, 'APISecrets', {
      secretName: DefaultIdBuilder.build('api-secrets'),
      description: 'API keys for FRED, World Bank, and Yahoo Finance',
      generateSecretString: {
        secretStringTemplate: JSON.stringify({
          FRED_API_KEY: '',
          WORLD_BANK_API_KEY: '',
          YAHOO_FINANCE_API_KEY: ''
        }),
        generateStringKey: 'password'
      }
    });

    // Create Python dependencies Lambda layer
    const pythonDependenciesLayer = new lambda.LayerVersion(this, 'PythonDependenciesLayer', {
      code: lambda.Code.fromAsset('../lambda/layers/python-dependencies'),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_10],
      description: 'Python dependencies for API collectors',
      layerVersionName: DefaultIdBuilder.build('python-dependencies-layer')
    });

    // Common Lambda configuration
    const lambdaConfig = {
      runtime: lambda.Runtime.PYTHON_3_10,
      timeout: Duration.minutes(15),
      memorySize: 1024,
      environment: {
        ENVIRONMENT: props.environment,
        BRONZE_BUCKET: props.bronzeBucket.bucketName,
        API_SECRETS_ARN: this.apiSecrets.secretArn
      }
    };

    // FRED API Data Collector
    this.fredCollector = new lambda.Function(this, 'FREDDataCollector', {
      ...lambdaConfig,
      handler: 'fred_collector.handler',
      code: lambda.Code.fromAsset('../lambda/api-collectors'),
      functionName: DefaultIdBuilder.build('fred-collector'),
      description: 'Collects economic indicators from FRED API',
      layers: [pythonDependenciesLayer],
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // World Bank API Data Collector
    this.worldBankCollector = new lambda.Function(this, 'WorldBankDataCollector', {
      ...lambdaConfig,
      handler: 'worldbank_collector.handler',
      code: lambda.Code.fromAsset('../lambda/api-collectors'),
      functionName: DefaultIdBuilder.build('worldbank-collector'),
      description: 'Collects economic indicators from World Bank API',
      layers: [pythonDependenciesLayer],
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Yahoo Finance API Data Collector
    this.yahooFinanceCollector = new lambda.Function(this, 'YahooFinanceDataCollector', {
      ...lambdaConfig,
      handler: 'yahoo_finance_collector.handler',
      code: lambda.Code.fromAsset('../lambda/api-collectors'),
      functionName: DefaultIdBuilder.build('yahoo-finance-collector'),
      description: 'Collects market data from Yahoo Finance API',
      layers: [pythonDependenciesLayer],
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Step Functions state machine for orchestrated data collection
    this.dataCollectionStateMachine = new sfn.StateMachine(this, 'APIDataCollectionWorkflow', {
      definition: sfn.Chain.start(new tasks.LambdaInvoke(this, 'CollectFREDData', {
        lambdaFunction: this.fredCollector,
        outputPath: '$.Payload',
        resultPath: '$.fred_data'
      }))
        .next(new tasks.LambdaInvoke(this, 'CollectWorldBankData', {
          lambdaFunction: this.worldBankCollector,
          outputPath: '$.Payload',
          resultPath: '$.worldbank_data'
        }))
        .next(new tasks.LambdaInvoke(this, 'CollectYahooFinanceData', {
          lambdaFunction: this.yahooFinanceCollector,
          outputPath: '$.Payload',
          resultPath: '$.yahoo_finance_data'
        }))
        .next(new sfn.Succeed(this, 'DataCollectionCompleted')),
      timeout: Duration.minutes(MACRO_CAUSAL_CONSTANTS.STEP_FUNCTIONS.EXECUTION_TIMEOUT_MINUTES),
      stateMachineName: DefaultIdBuilder.build('api-data-collection-workflow')
    });

    // EventBridge rule for scheduled data collection (daily at 8 AM UTC)
    this.scheduledCollectionRule = new events.Rule(this, 'ScheduledAPICollectionRule', {
      ruleName: DefaultIdBuilder.build('scheduled-api-collection-rule'),
      schedule: events.Schedule.cron({
        minute: '0',
        hour: '8', // 8 AM UTC
        day: '*',
        month: '*',
        year: '*'
      }),
      targets: [
        new targets.SfnStateMachine(this.dataCollectionStateMachine, {
          input: events.RuleTargetInput.fromObject({
            date: events.EventField.fromPath('$.time'),
            source: 'scheduled',
            collection_type: 'api_data'
          })
        })
      ]
    });

    // Grant permissions
    props.bronzeBucket.grantReadWrite(this.fredCollector);
    props.bronzeBucket.grantReadWrite(this.worldBankCollector);
    props.bronzeBucket.grantReadWrite(this.yahooFinanceCollector);

    // Grant Secrets Manager permissions
    this.apiSecrets.grantRead(this.fredCollector);
    this.apiSecrets.grantRead(this.worldBankCollector);
    this.apiSecrets.grantRead(this.yahooFinanceCollector);

    // Grant Step Functions permissions
    this.dataCollectionStateMachine.grantStartExecution(this.fredCollector);
    this.dataCollectionStateMachine.grantStartExecution(this.worldBankCollector);
    this.dataCollectionStateMachine.grantStartExecution(this.yahooFinanceCollector);

    // Add custom policies for API access
    const apiAccessPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'logs:DescribeLogGroups',
        'logs:DescribeLogStreams'
      ],
      resources: ['*']
    });

    this.fredCollector.addToRolePolicy(apiAccessPolicy);
    this.worldBankCollector.addToRolePolicy(apiAccessPolicy);
    this.yahooFinanceCollector.addToRolePolicy(apiAccessPolicy);
  }
}

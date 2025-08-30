import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';
import { SubnetType } from 'aws-cdk-lib/aws-ec2';
import { LambdaConfig } from '../configs/LambdaConfig';

export interface ApiDataCollectionProps {
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

  constructor(scope: Construct, id: string, props: ApiDataCollectionProps) {
    super(scope, id);

    // Create Python dependencies Lambda layer
    const pythonDependenciesLayerId = DefaultIdBuilder.build('python-dependencies-layer');
    const pythonDependenciesLayer = new lambda.LayerVersion(this, pythonDependenciesLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('python-dependencies')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python dependencies for API collectors', 
      layerVersionName: pythonDependenciesLayerId
    });

    const lambdaConfig = {
      runtime: LambdaConfig.DEFAULT_PYTHON_RUNTIME,
      timeout: Duration.minutes(15),
      memorySize: 1024
    };

    // FRED API Data Collector
    const fredCollectorLambdaFunctionId = DefaultIdBuilder.build('fred-collector');
    this.fredCollector = new lambda.Function(this, fredCollectorLambdaFunctionId, {
      ...lambdaConfig,
      handler: 'fred_collector.handler',
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonCodePath('api-collectors', 'fred_collector')),
      functionName: fredCollectorLambdaFunctionId,
      description: 'Collects economic indicators from FRED API',
      layers: [pythonDependenciesLayer],
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // World Bank API Data Collector
    const worldBankCollectorLambdaFunctionId = DefaultIdBuilder.build('worldbank-collector');
    this.worldBankCollector = new lambda.Function(this, worldBankCollectorLambdaFunctionId, {
      ...lambdaConfig,
      handler: 'worldbank_collector.handler',
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonCodePath('api-collectors', 'worldbank_collector')),
      functionName: worldBankCollectorLambdaFunctionId,
      description: 'Collects economic indicators from World Bank API',
      layers: [pythonDependenciesLayer],
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Yahoo Finance API Data Collector
    const yahooFinanceCollectorLambdaFunctionId = DefaultIdBuilder.build('yahoo-finance-collector');
    this.yahooFinanceCollector = new lambda.Function(this, yahooFinanceCollectorLambdaFunctionId, {
      ...lambdaConfig,
      handler: 'yahoo_finance_collector.handler',
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonCodePath('api-collectors', 'yahoo_finance_collector')),
      functionName: yahooFinanceCollectorLambdaFunctionId,
      description: 'Collects market data from Yahoo Finance API',
      layers: [pythonDependenciesLayer],
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Step Functions state machine for orchestrated data collection
    const dataCollectionStateMachineId = DefaultIdBuilder.build('api-data-collection-workflow');
    this.dataCollectionStateMachine = new sfn.StateMachine(this, dataCollectionStateMachineId, {
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
      stateMachineName: dataCollectionStateMachineId
    });

    // EventBridge rule for scheduled data collection (daily at 8 AM UTC)
    const scheduledCollectionRuleId = DefaultIdBuilder.build('scheduled-api-collection-rule');
    this.scheduledCollectionRule = new events.Rule(this, scheduledCollectionRuleId, {
      ruleName: scheduledCollectionRuleId,
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

    // Grant Step Functions permissions
    this.dataCollectionStateMachine.grantStartExecution(this.fredCollector);
    this.dataCollectionStateMachine.grantStartExecution(this.worldBankCollector);
    this.dataCollectionStateMachine.grantStartExecution(this.yahooFinanceCollector);
  }
}

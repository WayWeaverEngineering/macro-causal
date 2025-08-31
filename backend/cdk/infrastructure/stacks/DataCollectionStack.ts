import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import { Duration, Stack, StackProps } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';
import { LambdaConfig } from '../configs/LambdaConfig';
import { PythonLambdaLayersStack } from './PythonLambdaLayersStack';

export interface DataCollectionStackProps extends StackProps {
  bronzeBucket: s3.Bucket;
  pythonLambdaLayersStack: PythonLambdaLayersStack;
}

export class DataCollectionStack extends Stack {
  public readonly fredCollector: lambda.Function;
  public readonly worldBankCollector: lambda.Function;
  public readonly yahooFinanceCollector: lambda.Function;
  public readonly dataCollectionStateMachine: sfn.StateMachine;
  public readonly scheduledCollectionRule: events.Rule;

  constructor(scope: Construct, id: string, props: DataCollectionStackProps) {
    super(scope, id, props);

    const lambdaConfig = {
      runtime: LambdaConfig.DEFAULT_PYTHON_RUNTIME,
      timeout: Duration.minutes(15),
      memorySize: 1024
    };
    const dataCollectorsFolder = LambdaConfig.getLambdaPythonCodePath('data-collectors');

    // FRED API Data Collector
    const fredCollectorLambdaFunctionId = DefaultIdBuilder.build('fred-collector');
    this.fredCollector = new lambda.Function(this, fredCollectorLambdaFunctionId, {
      ...lambdaConfig,
      handler: 'fred_collector.handler',
      code: lambda.Code.fromAsset(dataCollectorsFolder),
      functionName: fredCollectorLambdaFunctionId,
      description: 'Collects economic indicators from FRED API',
      layers: [
        props.pythonLambdaLayersStack.botoLambdaLayer,
        props.pythonLambdaLayersStack.numpyLambdaLayer,
        props.pythonLambdaLayersStack.pandasLambdaLayer,
        props.pythonLambdaLayersStack.requestsLambdaLayer,
        props.pythonLambdaLayersStack.dateutilsLambdaLayer
      ]
    });

    // World Bank API Data Collector
    const worldBankCollectorLambdaFunctionId = DefaultIdBuilder.build('worldbank-collector');
    this.worldBankCollector = new lambda.Function(this, worldBankCollectorLambdaFunctionId, {
      ...lambdaConfig,
      handler: 'worldbank_collector.handler',
      code: lambda.Code.fromAsset(dataCollectorsFolder),
      functionName: worldBankCollectorLambdaFunctionId,
      description: 'Collects economic indicators from World Bank API',
      layers: [
        props.pythonLambdaLayersStack.botoLambdaLayer,
        props.pythonLambdaLayersStack.numpyLambdaLayer,
        props.pythonLambdaLayersStack.pandasLambdaLayer,
        props.pythonLambdaLayersStack.requestsLambdaLayer,
        props.pythonLambdaLayersStack.dateutilsLambdaLayer
      ]
    });

    // Yahoo Finance API Data Collector
    const yahooFinanceCollectorLambdaFunctionId = DefaultIdBuilder.build('yahoo-finance-collector');
    this.yahooFinanceCollector = new lambda.Function(this, yahooFinanceCollectorLambdaFunctionId, {
      ...lambdaConfig,
      handler: 'yahoo_finance_collector.handler',
      code: lambda.Code.fromAsset(dataCollectorsFolder),
      functionName: yahooFinanceCollectorLambdaFunctionId,
      description: 'Collects market data from Yahoo Finance API',
      layers: [
        props.pythonLambdaLayersStack.botoLambdaLayer,
        props.pythonLambdaLayersStack.numpyLambdaLayer,
        props.pythonLambdaLayersStack.pandasLambdaLayer,
        props.pythonLambdaLayersStack.requestsLambdaLayer,
        props.pythonLambdaLayersStack.dateutilsLambdaLayer,
      ]
    });

    // Define the state machine workflow
    const dataCollectionWorkflow =
      sfn.Chain.start(new tasks.LambdaInvoke(this, DefaultIdBuilder.build('collect-fred-data'), {
        lambdaFunction: this.fredCollector,
        outputPath: '$.Payload',
        resultPath: '$.fred_data'
      }))
      .next(new tasks.LambdaInvoke(this, DefaultIdBuilder.build('collect-worldbank-data'), {
        lambdaFunction: this.worldBankCollector,
        outputPath: '$.Payload',
        resultPath: '$.worldbank_data'
      }))
      .next(new tasks.LambdaInvoke(this, DefaultIdBuilder.build('collect-yahoo-finance-data'), {
        lambdaFunction: this.yahooFinanceCollector,
        outputPath: '$.Payload',
        resultPath: '$.yahoo_finance_data'
      }))
      .next(new sfn.Succeed(this, DefaultIdBuilder.build('data-collection-completed')));

    // Step Functions state machine for orchestrated data collection
    const dataCollectionStateMachineId = DefaultIdBuilder.build('data-collection-state-machine');
    this.dataCollectionStateMachine = new sfn.StateMachine(this, dataCollectionStateMachineId, {
      definitionBody: sfn.DefinitionBody.fromChainable(dataCollectionWorkflow),
      timeout: Duration.minutes(MACRO_CAUSAL_CONSTANTS.STEP_FUNCTIONS.EXECUTION_TIMEOUT_MINUTES),
      stateMachineName: dataCollectionStateMachineId
    });

    // EventBridge rule for scheduled data collection (daily at 8 AM UTC)
    const scheduledCollectionRuleId = DefaultIdBuilder.build('scheduled-data-collection-rule');
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
            collection_type: 'data'
          })
        })
      ]
    });

    // Grant permissions
    props.bronzeBucket.grantReadWrite(this.fredCollector);
    props.bronzeBucket.grantReadWrite(this.worldBankCollector);
    props.bronzeBucket.grantReadWrite(this.yahooFinanceCollector);
  }
}

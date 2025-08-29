import { Construct } from 'constructs';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Duration, RemovalPolicy } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS, RESOURCE_NAMES } from '../../utils/Constants';

export interface MLTrainingProps {
  environment: string;
  accountId: string;
  region: string;
  goldBucket: s3.Bucket;
  artifactsBucket: s3.Bucket;
  vpc?: ec2.IVpc;
}

export class MLTrainingConstruct extends Construct {
  public readonly eksCluster: eks.Cluster;
  public readonly trainingRepo: ecr.Repository;
  public readonly trainingRole: iam.Role;

  constructor(scope: Construct, id: string, props: MLTrainingProps) {
    super(scope, id);

    // VPC for EKS cluster
    const vpc = props.vpc || new ec2.Vpc(this, 'MLTrainingVPC', {
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: 'private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        }
      ]
    });

    // EKS cluster for ML training
    this.eksCluster = new eks.Cluster(this, RESOURCE_NAMES.EKS_CLUSTER, {
      version: eks.KubernetesVersion.of(MACRO_CAUSAL_CONSTANTS.EKS.KUBERNETES_VERSION),
      vpc: vpc,
      defaultCapacity: 0, // Use Karpenter for auto-scaling
      clusterName: DefaultIdBuilder.build('ml-training-cluster'),
      endpointAccess: eks.EndpointAccess.PUBLIC_AND_PRIVATE,
      vpcSubnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
      role: new iam.Role(this, 'EKSClusterRole', {
        assumedBy: new iam.ServicePrincipal('eks.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSClusterPolicy')
        ]
      }),
      kubectlLayer: new lambda.LayerVersion(this, 'KubectlLayer', {
        code: lambda.Code.fromAsset('kubectl-layer'),
        compatibleRuntimes: [lambda.Runtime.NODEJS_18_X],
        description: 'Kubectl layer for EKS cluster'
      })
    });

    // ECR repository for training images
    this.trainingRepo = new ecr.Repository(this, 'TrainingRepo', {
      repositoryName: DefaultIdBuilder.build('ml-training'),
      imageScanOnPush: true,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          maxImageCount: 10,
          rulePriority: 1
        }
      ]
    });

    // IAM role for Ray training jobs
    this.trainingRole = new iam.Role(this, 'TrainingRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSWorkerNodePolicy'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKS_CNI_Policy'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3ReadOnlyAccess')
      ]
    });

    // Grant access to S3 buckets
    props.goldBucket.grantRead(this.trainingRole);
    props.artifactsBucket.grantReadWrite(this.trainingRole);
    this.trainingRepo.grantPull(this.trainingRole);

    // Add custom policies for Ray
    this.trainingRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'ec2:DescribeInstances',
        'ec2:DescribeRegions',
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'logs:DescribeLogGroups',
        'logs:DescribeLogStreams'
      ],
      resources: ['*']
    }));


  }
}

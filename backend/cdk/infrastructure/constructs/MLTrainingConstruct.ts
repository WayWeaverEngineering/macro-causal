import { Construct } from 'constructs';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { RemovalPolicy } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';

export interface MLTrainingProps {  
  accountId: string;
  region: string;
  goldBucket: s3.Bucket;
  artifactsBucket: s3.Bucket;
  vpc: ec2.IVpc;
}

export class MLTrainingConstruct extends Construct {
  public readonly eksCluster: eks.Cluster;
  public readonly trainingRepo: ecr.Repository;
  public readonly trainingRole: iam.Role;

  constructor(scope: Construct, id: string, props: MLTrainingProps) {
    super(scope, id);

    // EKS cluster for ML training
    const eksClusterId = DefaultIdBuilder.build('ml-training-cluster');
    this.eksCluster = new eks.Cluster(this, eksClusterId, {
      version: eks.KubernetesVersion.of(MACRO_CAUSAL_CONSTANTS.EKS.KUBERNETES_VERSION),
      vpc: props.vpc,
      defaultCapacity: 0, // Use Karpenter for auto-scaling
      clusterName: eksClusterId,
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
    const trainingRepoId = DefaultIdBuilder.build('ml-training');
    this.trainingRepo = new ecr.Repository(this, trainingRepoId, {
      repositoryName: trainingRepoId,
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
    const trainingRoleId = DefaultIdBuilder.build('ml-training-role');
    this.trainingRole = new iam.Role(this, trainingRoleId, {
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

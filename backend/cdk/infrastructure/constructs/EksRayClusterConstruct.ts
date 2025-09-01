import { Construct } from 'constructs';
import { Stack } from 'aws-cdk-lib';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { KubectlV28Layer } from '@aws-cdk/lambda-layer-kubectl-v28';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';

export interface EksRayClusterProps {
  name: string;
  goldBucket: s3.Bucket;
  artifactsBucket: s3.Bucket;
  modelRegistryTable: string;
}

export class EksRayClusterConstruct extends Construct {
  public readonly cluster: eks.Cluster;
  public readonly rayNamespace: string;
  public readonly rayServiceAccount: iam.Role;

  constructor(scope: Construct, id: string, props: EksRayClusterProps) {
    super(scope, id);

    // Create EKS cluster
    const clusterId = DefaultIdBuilder.build('ray-cluster');
    this.cluster = new eks.Cluster(this, clusterId, {
      version: eks.KubernetesVersion.V1_28,
      defaultCapacity: 0, // We'll add node groups manually
      clusterName: `${props.name}-ray-cluster`,
      endpointAccess: eks.EndpointAccess.PUBLIC_AND_PRIVATE,
      kubectlLayer: new KubectlV28Layer(this, DefaultIdBuilder.build('kubectl-v28-layer')),
    });

    // Add system node group for control plane add-ons
    const systemNodeGroupId = DefaultIdBuilder.build('system-node-group');
    this.cluster.addNodegroupCapacity(systemNodeGroupId, {
      instanceTypes: [ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE)],
      minSize: 2,
      desiredSize: 2,
      maxSize: 4,
      labels: {
        'system': 'true',
      },
    });

    // Add Ray worker node group
    const rayNodeGroupId = DefaultIdBuilder.build('ray-node-group');
    this.cluster.addNodegroupCapacity(rayNodeGroupId, {
      instanceTypes: [
        ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.LARGE),
        ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.XLARGE),
      ],
      minSize: MACRO_CAUSAL_CONSTANTS.EKS.MIN_SIZE,
      maxSize: MACRO_CAUSAL_CONSTANTS.EKS.MAX_SIZE,
      desiredSize: MACRO_CAUSAL_CONSTANTS.EKS.DESIRED_SIZE,
      nodeRole: this.createNodeRole(),
      labels: {
        'ray.io/node-type': 'worker',
        'ray.io/cluster': props.name,
        'workload': 'ray-cpu',
      },
      taints: [], // No taints for now
    });

    // Create Ray namespace
    this.rayNamespace = 'ray';

    // Install KubeRay operator via Helm
    new eks.HelmChart(this, 'KubeRayOperator', {
      cluster: this.cluster,
      namespace: 'ray-system',
      createNamespace: true,
      repository: 'https://ray-project.github.io/kuberay-helm/',
      chart: 'kuberay-operator',
      version: '1.4.2',
    });

    // Create Ray service account with IRSA
    this.rayServiceAccount = this.createRayServiceAccount(props);

    // Grant S3 access to Ray service account
    props.goldBucket.grantRead(this.rayServiceAccount);
    props.artifactsBucket.grantReadWrite(this.rayServiceAccount);

    // Add DynamoDB permissions for model registry
    this.rayServiceAccount.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:PutItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem',
        'dynamodb:Query',
        'dynamodb:Scan',
      ],
      resources: [`arn:aws:dynamodb:${Stack.of(this).region}:${Stack.of(this).account}:table/${props.modelRegistryTable}`],
    }));

    // Add ECR permissions for pulling training images
    this.rayServiceAccount.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: ['*'],
    }));

    // Add CloudWatch permissions for logging
    this.rayServiceAccount.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'logs:DescribeLogGroups',
        'logs:DescribeLogStreams',
      ],
      resources: ['*'],
    }));

    // Create Ray cluster manifest
    this.createRayClusterManifest(props);
  }

  private createNodeRole(): iam.Role {
    const nodeRoleId = DefaultIdBuilder.build('ray-node-role');
    const nodeRole = new iam.Role(this, nodeRoleId, {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSWorkerNodePolicy'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKS_CNI_Policy'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    return nodeRole;
  }

  private createRayServiceAccount(props: EksRayClusterProps): iam.Role {
    const serviceAccountId = DefaultIdBuilder.build('ray-service-account');
    const serviceAccount = new iam.Role(this, serviceAccountId, {
      assumedBy: new iam.FederatedPrincipal(
        this.cluster.openIdConnectProvider.openIdConnectProviderArn,
        {
          StringEquals: {
            [`${this.cluster.openIdConnectProvider.openIdConnectProviderIssuer}:aud`]: 'sts.amazonaws.com',
            [`${this.cluster.openIdConnectProvider.openIdConnectProviderIssuer}:sub`]: `system:serviceaccount:${this.rayNamespace}:ray-service-account`,
          },
        },
        'sts:AssumeRoleWithWebIdentity'
      ),
    });

    return serviceAccount;
  }

  private createRayClusterManifest(props: EksRayClusterProps): void {
    // Create Ray cluster manifest
    const rayCluster = this.cluster.addManifest('RayCluster', {
      apiVersion: 'ray.io/v1',
      kind: 'RayCluster',
      metadata: { 
        name: `${props.name}-ray-cluster`, 
        namespace: this.rayNamespace 
      },
      spec: {
        rayVersion: '2.9.0',
        enableInTreeAutoscaling: true,
        headGroupSpec: {
          serviceType: 'ClusterIP',
          template: {
            spec: {
              serviceAccountName: 'ray-service-account',
              containers: [{
                name: 'ray-head',
                image: 'public.ecr.aws/rayproject/ray:2.9.0-py310',
                resources: {
                  requests: { 
                    cpu: MACRO_CAUSAL_CONSTANTS.RAY.HEAD_CPU, 
                    memory: MACRO_CAUSAL_CONSTANTS.RAY.HEAD_MEMORY 
                  },
                  limits: { 
                    cpu: MACRO_CAUSAL_CONSTANTS.RAY.HEAD_CPU, 
                    memory: MACRO_CAUSAL_CONSTANTS.RAY.HEAD_MEMORY 
                  }
                },
                env: [
                  { name: 'RAY_ENABLE_WINDOWS_ERROR_REPORTING', value: '0' },
                  { name: 'AWS_REGION', value: Stack.of(this).region }
                ]
              }],
              nodeSelector: { 'workload': 'ray-cpu' }
            }
          }
        },
        workerGroupSpecs: [{
          groupName: 'workers',
          replicas: MACRO_CAUSAL_CONSTANTS.RAY.MIN_WORKERS,
          minReplicas: MACRO_CAUSAL_CONSTANTS.RAY.MIN_WORKERS,
          maxReplicas: MACRO_CAUSAL_CONSTANTS.RAY.MAX_WORKERS,
          rayStartParams: { 'num-cpus': MACRO_CAUSAL_CONSTANTS.RAY.WORKER_CPU },
          template: {
            spec: {
              serviceAccountName: 'ray-service-account',
              containers: [{
                name: 'ray-worker',
                image: 'public.ecr.aws/rayproject/ray:2.9.0-py310',
                resources: {
                  requests: { 
                    cpu: MACRO_CAUSAL_CONSTANTS.RAY.WORKER_CPU, 
                    memory: MACRO_CAUSAL_CONSTANTS.RAY.WORKER_MEMORY 
                  },
                  limits: { 
                    cpu: MACRO_CAUSAL_CONSTANTS.RAY.WORKER_CPU, 
                    memory: MACRO_CAUSAL_CONSTANTS.RAY.WORKER_MEMORY 
                  }
                }
              }],
              nodeSelector: { 'workload': 'ray-cpu' }
            }
          }
        }]
      }
    });

    // Create Ray service account in Kubernetes
    const rayServiceAccount = this.cluster.addManifest('RayServiceAccount', {
      apiVersion: 'v1',
      kind: 'ServiceAccount',
      metadata: {
        name: 'ray-service-account',
        namespace: this.rayNamespace,
        annotations: {
          'eks.amazonaws.com/role-arn': this.rayServiceAccount.roleArn,
        },
      },
    });

    // Create Ray namespace
    const rayNamespace = this.cluster.addManifest('RayNamespace', {
      apiVersion: 'v1',
      kind: 'Namespace',
      metadata: {
        name: this.rayNamespace,
      },
    });

    // Set dependencies
    rayCluster.node.addDependency(rayServiceAccount);
    rayCluster.node.addDependency(rayNamespace);
  }
}

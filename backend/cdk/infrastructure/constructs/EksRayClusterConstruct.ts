import { Construct } from 'constructs';
import { Stack } from 'aws-cdk-lib';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { KubectlV28Layer } from '@aws-cdk/lambda-layer-kubectl-v28';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';

export interface EksRayClusterProps {
  name: string;
  goldBucket: s3.Bucket;
  artifactsBucket: s3.Bucket;
  modelRegistryTable: dynamodb.Table;
}

export class EksRayClusterConstruct extends Construct {
  public readonly cluster: eks.Cluster;
  public readonly rayNamespace: string;
  public readonly rayServiceAccount: iam.Role;

  constructor(scope: Construct, id: string, props: EksRayClusterProps) {
    super(scope, id);

    // Create VPC with 2 AZs and no NAT gateways for cost efficiency
    // EKS requires subnets from at least two AZs
    const eksClusterVpcId = DefaultIdBuilder.build('eks-cluster-vpc');
    const vpc = new ec2.Vpc(this, eksClusterVpcId, {
      ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/21'), // Keep VPC small; adjust if you need more IPs
      maxAzs: 2,          // CRITICAL: EKS needs at least 2 AZs
      natGateways: 0,     // no NATs, no extra EIPs
      subnetConfiguration: [
        {
          name: 'public-a-b',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,   // smaller public subnets to conserve IPs
        },
        // Optional: add isolated subnets later if needed
        // {
        //   name: 'isolated-a-b',
        //   subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
        //   cidrMask: 24,
        // },
      ],
    });

    // Create EKS cluster using public subnets across both AZs
    const clusterId = DefaultIdBuilder.build('ray-cluster');
    this.cluster = new eks.Cluster(this, clusterId, {
      vpc,
      version: eks.KubernetesVersion.V1_28,
      defaultCapacity: 0,
      clusterName: `${props.name}-ray-cluster`,
      // Public endpoint: reachable from internet and from within VPC
      endpointAccess: eks.EndpointAccess.PUBLIC_AND_PRIVATE,
      kubectlLayer: new KubectlV28Layer(this, DefaultIdBuilder.build('kubectl-v28-layer')),
      // Ensure all cluster-managed resources land in public subnets
      vpcSubnets: [{ subnetType: ec2.SubnetType.PUBLIC }], // spans both AZs automatically
    });

    // Add system node group for control plane add-ons in public subnets
    const systemNodeGroupId = DefaultIdBuilder.build('system-node-group');
    this.cluster.addNodegroupCapacity(systemNodeGroupId, {
      subnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceTypes: [ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE)],
      minSize: 1,
      desiredSize: 1,
      maxSize: 2,
      labels: { 'system': 'true' },
    });

    // Add Ray worker node group in public subnets
    const rayNodeGroupId = DefaultIdBuilder.build('ray-node-group');
    this.cluster.addNodegroupCapacity(rayNodeGroupId, {
      subnets: { subnetType: ec2.SubnetType.PUBLIC },
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
      taints: [], // keep as-is for now
    });

    // Create Ray namespace string stays as-is
    this.rayNamespace = 'ray';

    // Create the Kubernetes Namespace manifest (move it here from createRayClusterManifest)
    const rayNamespace = this.cluster.addManifest('RayNamespace', {
      apiVersion: 'v1',
      kind: 'Namespace',
      metadata: { name: this.rayNamespace },
    });

    // Install KubeRay operator via Helm
    new eks.HelmChart(this, 'KubeRayOperator', {
      cluster: this.cluster,
      namespace: 'ray-system',
      createNamespace: true,
      repository: 'https://ray-project.github.io/kuberay-helm/',
      chart: 'kuberay-operator',
      version: '1.4.2',
    });

    // Create ServiceAccount with IRSA handled by CDK
    const raySa = this.cluster.addServiceAccount('RaySa', {
      name: 'ray-service-account',
      namespace: this.rayNamespace,
    });
    raySa.node.addDependency(rayNamespace); // ensure NS exists before SA

    // Keep the field name; assign the IAM Role behind the SA
    this.rayServiceAccount = raySa.role as iam.Role;

    // Grant S3 access to Ray service account
    props.goldBucket.grantRead(raySa.role);
    props.artifactsBucket.grantReadWrite(raySa.role);

    // Add DynamoDB permissions for model registry
    (raySa.role as iam.Role).addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:PutItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem',
        'dynamodb:Query',
        'dynamodb:Scan',
      ],
      resources: [props.modelRegistryTable.tableArn],
    }));

    // Add ECR permissions for pulling training images
    (raySa.role as iam.Role).addToPolicy(new iam.PolicyStatement({
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
    (raySa.role as iam.Role).addToPolicy(new iam.PolicyStatement({
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
    this.createRayClusterManifest(props, rayNamespace, raySa);
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

  private createRayClusterManifest(props: EksRayClusterProps, rayNamespace: eks.KubernetesManifest, raySa: eks.ServiceAccount): void {
    // Create Ray cluster manifest
    const rayCluster = this.cluster.addManifest('RayCluster', {
      apiVersion: 'ray.io/v1',
      kind: 'RayCluster',
      metadata: { 
        name: `${props.name}-ray-cluster`, 
        namespace: this.rayNamespace 
      },
      spec: {
        rayVersion: MACRO_CAUSAL_CONSTANTS.RAY.VERSION,
        enableInTreeAutoscaling: true,
        headGroupSpec: {
          serviceType: 'ClusterIP',
          template: {
            spec: {
              serviceAccountName: 'ray-service-account',
              containers: [{
                name: 'ray-head',
                image: `public.ecr.aws/rayproject/ray:${MACRO_CAUSAL_CONSTANTS.RAY.VERSION}-py310`,
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
                image: `public.ecr.aws/rayproject/ray:${MACRO_CAUSAL_CONSTANTS.RAY.VERSION}-py310`,
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

    // Ensure ordering
    rayCluster.node.addDependency(rayNamespace);
    rayCluster.node.addDependency(raySa);
  }
}

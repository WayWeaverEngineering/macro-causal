import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS, RESOURCE_NAMES } from '../../utils/Constants';

export interface VPCProps {
  environment: string;
  accountId: string;
  region: string;
  maxAzs?: number;
  natGateways?: number;
}

export class VPCConstruct extends Construct {
  public readonly vpc: ec2.Vpc;
  public readonly securityGroup: ec2.SecurityGroup;
  public readonly vpcEndpointS3: ec2.GatewayVpcEndpoint;
  public readonly vpcEndpointECR: ec2.InterfaceVpcEndpoint;
  public readonly vpcEndpointECRDocker: ec2.InterfaceVpcEndpoint;
  public readonly vpcEndpointSecretsManager: ec2.InterfaceVpcEndpoint;
  public readonly vpcEndpointSSM: ec2.InterfaceVpcEndpoint;

  constructor(scope: Construct, id: string, props: VPCProps) {
    super(scope, id);

    // Create VPC with public and private subnets
    this.vpc = new ec2.Vpc(this, RESOURCE_NAMES.VPC, {
      maxAzs: props.maxAzs || 2,
      natGateways: props.natGateways || 1,
      ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
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
        },
        {
          cidrMask: 24,
          name: 'isolated',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
        }
      ],
      enableDnsHostnames: true,
      enableDnsSupport: true,
      vpcName: DefaultIdBuilder.build('vpc')
    });

    // Create security group for the VPC
    this.securityGroup = new ec2.SecurityGroup(this, 'VPCSecurityGroup', {
      vpc: this.vpc,
      description: 'Security group for Macro Causal VPC',
      allowAllOutbound: true,
      securityGroupName: DefaultIdBuilder.build('vpc-sg')
    });

    // Allow HTTPS traffic for API calls
    this.securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(443),
      'Allow HTTPS traffic'
    );

    // Allow HTTP traffic for health checks
    this.securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'Allow HTTP traffic'
    );

    // VPC Endpoint for S3 (Gateway endpoint)
    this.vpcEndpointS3 = this.vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
      subnets: [
        { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        { subnetType: ec2.SubnetType.PRIVATE_ISOLATED }
      ]
    });

    // VPC Endpoint for ECR (Interface endpoint)
    this.vpcEndpointECR = this.vpc.addInterfaceEndpoint('ECREndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.ECR,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.securityGroup]
    });

    // VPC Endpoint for ECR Docker (Interface endpoint)
    this.vpcEndpointECRDocker = this.vpc.addInterfaceEndpoint('ECRDockerEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.securityGroup]
    });

    // VPC Endpoint for Secrets Manager (Interface endpoint)
    this.vpcEndpointSecretsManager = this.vpc.addInterfaceEndpoint('SecretsManagerEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.securityGroup]
    });

    // VPC Endpoint for Systems Manager (Interface endpoint)
    this.vpcEndpointSSM = this.vpc.addInterfaceEndpoint('SSMEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.SSM,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.securityGroup]
    });

    // Add tags to VPC resources
    this.vpc.node.addDependency(this.securityGroup);
  }
}

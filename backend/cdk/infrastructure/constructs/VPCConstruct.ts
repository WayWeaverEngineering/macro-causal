import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface VPCProps {
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
    const vpcId = DefaultIdBuilder.build('vpc');
    this.vpc = new ec2.Vpc(this, vpcId, {
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
      vpcName: vpcId
    });

    // Create security group for the VPC
    const securityGroupId = DefaultIdBuilder.build('vpc-sg');
    this.securityGroup = new ec2.SecurityGroup(this, securityGroupId, {
      vpc: this.vpc,
      description: 'Security group for Macro Causal VPC',
      allowAllOutbound: true,
      securityGroupName: securityGroupId
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
    const s3EndpointId = DefaultIdBuilder.build('s3-endpoint');
    this.vpcEndpointS3 = this.vpc.addGatewayEndpoint(s3EndpointId, {
      service: ec2.GatewayVpcEndpointAwsService.S3,
      subnets: [
        { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        { subnetType: ec2.SubnetType.PRIVATE_ISOLATED }
      ]
    });

    // VPC Endpoint for ECR (Interface endpoint)
    const ecrEndpointId = DefaultIdBuilder.build('ecr-endpoint');
    this.vpcEndpointECR = this.vpc.addInterfaceEndpoint(ecrEndpointId, {
      service: ec2.InterfaceVpcEndpointAwsService.ECR,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.securityGroup]
    });

    // VPC Endpoint for ECR Docker (Interface endpoint)
    const ecrDockerEndpointId = DefaultIdBuilder.build('ecr-docker-endpoint');
    this.vpcEndpointECRDocker = this.vpc.addInterfaceEndpoint(ecrDockerEndpointId, {
      service: ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.securityGroup]
    });

    // VPC Endpoint for Secrets Manager (Interface endpoint)
    const secretsManagerEndpointId = DefaultIdBuilder.build('secrets-manager-endpoint');
    this.vpcEndpointSecretsManager = this.vpc.addInterfaceEndpoint(secretsManagerEndpointId, {
      service: ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.securityGroup]
    });

    // VPC Endpoint for Systems Manager (Interface endpoint)
    const ssmEndpointId = DefaultIdBuilder.build('ssm-endpoint');
    this.vpcEndpointSSM = this.vpc.addInterfaceEndpoint(ssmEndpointId, {
      service: ec2.InterfaceVpcEndpointAwsService.SSM,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.securityGroup]
    });

    // Add tags to VPC resources
    this.vpc.node.addDependency(this.securityGroup);
  }
}

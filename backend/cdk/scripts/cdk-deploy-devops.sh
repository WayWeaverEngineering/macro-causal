#!/bin/bash
set -euo pipefail

# Load AWS_PROFILE from .env.aws if it exists
if [ -f .env.aws ]; then
  export $(grep AWS_PROFILE .env.aws | xargs)
fi

# Default to 'default' profile if not defined
AWS_PROFILE="${AWS_PROFILE:-default}"

echo "Synthesizing CDK app for devops environment..."
# Run the CDK app directly (same entrypoint as cdk.json) to synthesize
# the cloud assembly into cdk.out/devops
npx ts-node --prefer-ts-exts app/MacroCausal.ts

echo "Deploying devops CDK stacks using profile: $AWS_PROFILE"

# Deploy from the devops cloud assembly directory
# CdkAppBuild is configured to write the devops environment to cdk.out/devops
if [ -d "cdk.out/devops" ]; then
  npx cdk deploy --all --app "cdk.out/devops" --profile "$AWS_PROFILE"
else
  echo "Error: cdk.out/devops directory not found. Synthesis may have failed."
  exit 1
fi


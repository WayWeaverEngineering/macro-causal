#!/bin/bash
set -euo pipefail

# Load AWS_PROFILE from .env.aws if it exists
if [ -f .env.aws ]; then
  export $(grep AWS_PROFILE .env.aws | xargs)
fi

# Default to 'default' profile if not defined
AWS_PROFILE="${AWS_PROFILE:-default}"

echo "Synthesizing CDK app for harry-finance environment..."
# Run the CDK app directly (same entrypoint as cdk.json) to synthesize
# only the harry-finance environment into cdk.out/harry-finance
SYNTH_ENVIRONMENT=harry-finance npx ts-node --prefer-ts-exts app/MacroCausal.ts

echo "Deploying harry-finance CDK stacks using profile: $AWS_PROFILE"

# Deploy from the harry-finance cloud assembly directory
# CdkAppBuild is configured to write the harry-finance environment to cdk.out/harry-finance
if [ -d "cdk.out/harry-finance" ]; then
  npx cdk deploy --all --app "cdk.out/harry-finance" --profile "$AWS_PROFILE"
else
  echo "Error: cdk.out/harry-finance directory not found. Synthesis may have failed."
  exit 1
fi
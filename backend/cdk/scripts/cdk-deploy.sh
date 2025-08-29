#!/bin/bash
set -euo pipefail

# Load AWS_PROFILE from .env.aws if it exists
if [ -f .env.aws ]; then
  export $(grep AWS_PROFILE .env.aws | xargs)
fi

# Default to 'default' profile if not defined
AWS_PROFILE="${AWS_PROFILE:-default}"

echo "Deploying CDK stacks using profile: $AWS_PROFILE"

npx cdk deploy --all --profile "$AWS_PROFILE"

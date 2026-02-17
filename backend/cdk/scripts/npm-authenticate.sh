#!/bin/bash
set -euo pipefail

# Avoid "unbound variable" errors
: "${CI:=}"
: "${CODEBUILD_BUILD_ID:=}"

# === Load config from .env.aws if available ===
if [ -f .env.aws ]; then
  echo "Loading configuration from .env.aws..."
  export $(grep -v '^#' .env.aws | xargs)
fi

# === Configurable Constants ===
DOMAIN="wayweaver-shared-artifacts"
DOMAIN_OWNER="715067592333"
REPO="npm-typescript"
SCOPE="@wayweaver"
REGION="${AWS_DEPLOYMENT_REGION:-ap-southeast-1}"

echo "Authenticating with AWS CodeArtifact..."

# === Detect CI/CD Environment ===
if [ -n "$CI" ] || [ -n "$CODEBUILD_BUILD_ID" ]; then
  echo "CI/CD environment detected (CI=$CI, CODEBUILD_BUILD_ID=$CODEBUILD_BUILD_ID)"
  echo "Skipping .env.aws and AWS SSO checks. Assuming IAM role is preconfigured."

  # Unset AWS_PROFILE to ensure AWS CLI uses IAM role credentials instead of profile
  unset AWS_PROFILE

  export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
  export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
  export AWS_SESSION_TOKEN="${AWS_SESSION_TOKEN:-}"

  echo "AWS Identity:"
  aws sts get-caller-identity || {
    echo "Unable to get caller identity. Check IAM role or environment variables."
    exit 1
  }
else
  echo "Local development environment detected"

  if [ -f .env.aws ]; then
    export $(grep AWS_PROFILE .env.aws | xargs)
  fi

  AWS_PROFILE="${AWS_PROFILE:-infra-dev}"
  export AWS_PROFILE

  if ! aws --profile "$AWS_PROFILE" sts get-caller-identity &> /dev/null; then
    echo "AWS credentials not found or expired. Please execute './scripts/aws-sso-authenticate.sh' first."
    exit 1
  fi

  echo "AWS credentials valid for profile: $AWS_PROFILE"
fi

# === Get Auth Token and Registry URL ===
AUTH_TOKEN=$(aws codeartifact get-authorization-token \
  --domain "$DOMAIN" \
  --domain-owner "$DOMAIN_OWNER" \
  --region "$REGION" \
  --query authorizationToken \
  --output text)

REGISTRY_URL=$(aws codeartifact get-repository-endpoint \
  --domain "$DOMAIN" \
  --domain-owner "$DOMAIN_OWNER" \
  --repository "$REPO" \
  --format npm \
  --region "$REGION" \
  --query repositoryEndpoint \
  --output text | sed 's:/*$::')

export REGISTRY_URL

REGISTRY_HOST=$(echo "$REGISTRY_URL" | sed -E 's|^https://||' | sed 's:/*$::')

# === Determine actual userconfig path and set it ===
USER_NPMRC=$(npm config get userconfig)
export NPM_CONFIG_USERCONFIG="$USER_NPMRC"

# === Configure npm using safe method ===
# VERY IMPORTANT: From AWS Documentation:
# The registry URL must end with a forward slash (/). Otherwise, you cannot connect to the repository.
# https://docs.aws.amazon.com/codeartifact/latest/ug/npm-auth.html
npm config set "$SCOPE:registry" "$REGISTRY_URL/"
npm config set "//$REGISTRY_HOST/:_authToken" "$AUTH_TOKEN"

echo "npm config written to: $USER_NPMRC"
echo "Verifying npm auth setup..."

# Show registry and masked token
grep '^@wayweaver:registry=' "$USER_NPMRC" || echo "Registry not set"

grep ':_authToken=' "$USER_NPMRC" | while read -r line; do
  token=$(echo "$line" | cut -d '=' -f2)
  echo "Token (prefix only): ${token:0:3}******"
done

echo "npm is now authenticated with CodeArtifact"

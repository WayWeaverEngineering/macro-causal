#!/bin/bash

set -euo pipefail

# === Load config from .env.aws if available ===
if [ -f .env.aws ]; then
  echo "Loading configuration from .env.aws..."
  export $(grep -v '^#' .env.aws | xargs)
fi

# === Configurable Defaults (fallbacks if .env.aws not set) ===
PROFILE_NAME="${AWS_PROFILE:-infra-dev}"
SSO_SESSION_NAME="${AWS_SSO_SESSION:-infra-dev-sso-session}"
SSO_START_URL="${AWS_SSO_START_URL:-https://d-9067d3aa2c.awsapps.com/start}"
SSO_REGION="${AWS_SSO_REGION:-us-east-1}"
CLI_REGION="${AWS_CLI_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-715067592333}"
ROLE_NAME="${AWS_ROLE_NAME:-InfrastructureDeveloper}"


echo "Setting up AWS SSO profile '$PROFILE_NAME'..."

# Step 1: Configure the SSO session (replaces older direct profile SSO setup)
aws configure set sso_session "$SSO_SESSION_NAME" --profile "$PROFILE_NAME"
aws configure set sso_account_id "$ACCOUNT_ID" --profile "$PROFILE_NAME"
aws configure set sso_role_name "$ROLE_NAME" --profile "$PROFILE_NAME"
aws configure set region "$CLI_REGION" --profile "$PROFILE_NAME"

# Step 2: Write sso-session block manually (overwrite if needed)
CONFIG_FILE="$HOME/.aws/config"
SESSION_BLOCK="[sso-session $SSO_SESSION_NAME]"

# Remove any existing session block
echo "Cleaning up old [$SESSION_BLOCK] block if present..."
TMP_FILE=$(mktemp)

awk -v session="sso-session $SSO_SESSION_NAME" '
BEGIN { in_block = 0 }
/^\[.*\]$/ {
  if ($0 == "[" session "]") {
    in_block = 1
    next
  } else {
    in_block = 0
  }
}
!in_block { print }
' "$CONFIG_FILE" > "$TMP_FILE" && mv "$TMP_FILE" "$CONFIG_FILE"

# Append correct session config
echo "Writing $SESSION_BLOCK to ~/.aws/config..."
cat <<EOF >> "$CONFIG_FILE"
${SESSION_BLOCK}
sso_start_url = $SSO_START_URL
sso_region = $SSO_REGION
EOF

echo "SSO profile configured."
echo "Now logging in..."
aws sso login --profile "$PROFILE_NAME"

# Export AWS_PROFILE for current shell session
export AWS_PROFILE="$PROFILE_NAME"

echo "All done! Your AWS profile '$PROFILE_NAME' is now authenticated and ready to use."
echo "AWS_PROFILE has been exported to '$PROFILE_NAME' for this shell session."
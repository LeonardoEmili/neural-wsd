#!/usr/bin/env bash
# Script to download asset file from tag release using GitHub API v3.
# This code is mainly an adaptation to our env of the following stackoverflow thread.
# See: http://stackoverflow.com/a/35688093/55075    

# Validate settings.
GITHUB_API_TOKEN=`grep 'GITHUB_PERSONAL_ACCESS_TOKEN' ../.env | tr '=' '\n' | grep -v 'GITHUB_PERSONAL_ACCESS_TOKEN'`
repo=`grep 'GITHUB_REPO_NAME' ../.env | tr '=' '\n' | grep -v 'GITHUB_REPO_NAME'`
owner=`grep 'GITHUB_REPO_OWNER' ../.env | tr '=' '\n' | grep -v 'GITHUB_REPO_OWNER'`
tag=`grep 'GITHUB_REPO_RELEASE_TAG' ../.env | tr '=' '\n' | grep -v 'GITHUB_REPO_RELEASE_TAG'`

[ !"$GITHUB_API_TOKEN" ] || { echo "Error: Please define GITHUB_API_TOKEN variable." >&2; exit 1; }
[ !"$repo" ] || { echo "Error: Please define GITHUB_REPO_NAME variable." >&2; exit 1; }
[ !"$owner" ] || { echo "Error: Please define GITHUB_REPO_OWNER variable." >&2; exit 1; }
[ !"$tag" ] || { echo "Error: Please define GITHUB_REPO_RELEASE_TAG variable." >&2; exit 1; }

# Define variables.
GH_API="https://api.github.com"
GH_REPO="$GH_API/repos/$owner/$repo"
GH_TAGS="$GH_REPO/releases/tags/$tag"
AUTH="Authorization: token $GITHUB_API_TOKEN"
WGET_ARGS="--content-disposition --auth-no-challenge --no-cookie"
CURL_ARGS="-LJO#"

# Define asset name and output path
name=$1
output_path=$2

# Validate token.
curl -o /dev/null -sH "$AUTH" $GH_REPO || { echo "Error: Invalid repo, token or network issue!";  exit 1; }

# Read asset tags.
response=$(curl -sH "$AUTH" $GH_TAGS)
# Get ID of the asset based on given name.
eval $(echo "$response" | grep -C3 "name.:.\+$name" | grep -w id | tr : = | tr -cd '[[:alnum:]]=')
[ "$id" ] || { echo "Error: Failed to get asset id, response: $response" | awk 'length($0)<100' >&2; exit 1; }
GH_ASSET="$GH_REPO/releases/assets/$id"

# Download asset file.
echo "Downloading asset..." >&2
curl -o "$output_path" $CURL_ARGS -H "Authorization: token $GITHUB_API_TOKEN" -H 'Accept: application/octet-stream' "$GH_ASSET"
echo "$0 done." >&2
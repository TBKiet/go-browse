#!/bin/bash

HOST=$BASE_URL
PORT="8023"

# Construct the full URL
FULL_URL="${HOST}:${PORT}"

# Stop and remove the gitlab container if it exists
if [ "$(docker ps -q -f name=gitlab)" ]; then
    echo "Stopping and removing existing gitlab container..."
    docker stop gitlab
    docker rm gitlab
fi

echo "Building the gitlab image..."
docker run --name gitlab -d -p ${PORT}:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start

echo "Waiting for the gitlab container to start..."
# Wait for the gitlab container to start
sleep 30

echo "Configuring the gitlab container..."
docker exec gitlab sed -i "s|^external_url.*|external_url '${FULL_URL}'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure

curl "$FULL_URL"

# Wait for GitLab to be fully initialized with a healthcheck
echo "Waiting for GitLab to be fully ready..."
MAX_RETRIES=30
COUNT=0
DELAY=10

while [ $COUNT -lt $MAX_RETRIES ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$FULL_URL/users/sign_in")
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "GitLab is ready! (Status code: $HTTP_CODE)"
        break
    else
        echo "GitLab not ready yet. Status code: $HTTP_CODE. Retrying in ${DELAY}s..."
        sleep $DELAY
        COUNT=$((COUNT+1))
    fi
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo "Warning: Reached maximum retries. GitLab may not be fully initialized."
fi

# Additional wait time to be safe
sleep 60
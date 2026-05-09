#!/bin/bash

HOST=$BASE_URL
PORT="9999"

# Construct the full URL
FULL_URL="${HOST}:${PORT}"

# Stop and remove the reddit container if it exists
if [ "$(docker ps -q -f name=forum)" ]; then
    echo "Stopping and removing existing forum container..."
    docker stop forum
    docker rm forum
fi

echo "Building the forum image..."
docker run --name forum -p ${PORT}:80 -d postmill-populated-exposed-withimg

# Wait for the forum container to start
echo "Waiting for the forum container to start..."
sleep 60

echo "Configuring the forum container..."
docker exec forum sed -i "s/^ENABLE_EXPERIMENTAL_REST_API.*/ENABLE_EXPERIMENTAL_REST_API=1/" .env
docker exec -it forum psql -U postmill -d postmill -c "UPDATE users SET trusted = true WHERE username = 'MarvelsGrantMan136';"

curl "$FULL_URL"

# Additional wait time to be safe
sleep 30
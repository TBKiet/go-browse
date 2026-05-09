#!/bin/bash

HOST=$BASE_URL
PORT="7770"

# Construct the full URL
FULL_URL="${HOST}:${PORT}"

# Stop and remove the shopping container if it exists
if [ "$(docker ps -q -f name=shopping)" ]; then
    echo "Stopping and removing existing shopping container..."
    docker stop shopping
    docker rm shopping
fi


echo "Building the shopping image..."
docker run --name shopping -p ${PORT}:80 -d shopping_final_0712

# Wait for the shopping container to start
echo "Waiting for the shopping container to start..."
sleep 30


echo "Configuring the shopping container..."
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="${FULL_URL}" # no trailing /
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value=\"${FULL_URL}/\" WHERE path = \"web/secure/base_url\";"
docker exec shopping /var/www/magento2/bin/magento cache:flush

curl "$FULL_URL"

# Additional wait time to be safe
sleep 30
#!/bin/bash

HOST=$BASE_URL
PORT="7780"

# Construct the full URL
FULL_URL="${HOST}:${PORT}"

# Stop and remove the shopping admin container if it exists
if [ "$(docker ps -q -f name=shopping-admin)" ]; then
    echo "Stopping and removing existing shopping admin container..."
    docker stop shopping-admin
    docker rm shopping-admin
fi

echo "Building the shopping admin image..."
docker run --name shopping_admin -p ${PORT}:80 -d shopping_admin_final_0719

# Wait for the shopping admin container to start
echo "Waiting for the shopping admin container to start..."
sleep 30

echo "Configuring the shopping admin container..."
# remove the requirement to reset password
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="${FULL_URL}" # no trailing /
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value=\"${FULL_URL}/\" WHERE path = \"web/secure/base_url\";"
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

curl "$FULL_URL/admin"

# Additional wait time to be safe
sleep 30
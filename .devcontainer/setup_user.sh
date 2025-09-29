#!/bin/bash
set -e

if id --user $USERNAME >/dev/null 2>&1; then
    echo "user:$USERNAME already exists, skipping setup ..."
else
    groupadd --gid $USER_GID $USERNAME
    useradd --uid $USER_UID --gid $USER_GID --create-home $USERNAME
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    setfacl --modify user:$USERNAME:--x /root
    setfacl --recursive --modify user:$USERNAME:rwx /root/.vscode-server
    setfacl --recursive --default --modify user:$USERNAME:rwx /root/.vscode-server
fi

exit 0

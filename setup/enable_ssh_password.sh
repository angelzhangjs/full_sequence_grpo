#!/bin/bash
# Enable SSH password authentication

sudo sed -i.bak 's/^PasswordAuthentication no$/PasswordAuthentication yes/' /etc/ssh/sshd_config.d/50-cloud-init.conf
sudo sed -i.bak 's/^PasswordAuthentication no$/PasswordAuthentication yes/' /etc/ssh/sshd_config.d/60-cloudimg-settings.conf

echo "SSH password authentication enabled"
echo "Restarting SSH service..."
sudo systemctl restart sshd

echo "Done!"


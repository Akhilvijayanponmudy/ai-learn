# FTP/SFTP Setup Guide

## Where to configure
Go to **WooCommerce → Import Export → Settings → FTP/SFTP**.

## FTP vs SFTP
- FTP: older, unencrypted
- SFTP: recommended, encrypted (runs over SSH)

## Required fields
### FTP
- Host
- Port (default 21)
- Username
- Password
- Remote directory path

### SFTP
- Host
- Port (default 22)
- Username
- Password or SSH key (if supported)
- Remote directory path

## Common errors
### Authentication failed
- Recheck username/password
- Confirm the server allows password login
- Confirm the port (22 for SFTP)

### Connection timeout
- Server is not reachable from your hosting environment
- Firewall blocks outbound connections
- Wrong host or port

### Permission denied
- Remote directory does not exist
- User lacks write permissions
- Try exporting to a directory you own, like `/uploads/exports/`

## Tips
- Always test with a small export first.
- Prefer SFTP over FTP.
- Avoid spaces in remote folder names.

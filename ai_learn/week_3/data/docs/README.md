# WebToffee Import Export - Quick Start

## What this plugin does
This plugin lets you import and export WooCommerce data (Orders, Products, Customers/Users, Coupons) using CSV files.
It also supports scheduled exports and optional FTP/SFTP file transfers.

## Common workflows
### Export orders
1. Go to **WooCommerce → Import Export**.
2. Choose **Orders**.
3. Select export filters like date range, status, payment method.
4. Run export and download the CSV.

### Import products
1. Go to **WooCommerce → Import Export**.
2. Choose **Products → Import**.
3. Upload a CSV and map columns.
4. Run import.

## Troubleshooting checklist
- If export/import is slow, try smaller batches.
- Check PHP memory limit and max execution time.
- If using FTP/SFTP, verify credentials and server connectivity.
- If you see “permission denied”, check directory permissions on the remote server.

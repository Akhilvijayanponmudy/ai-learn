# FAQ

## Export is empty
- Check filters (date range/status)
- Verify there are matching records
- Try removing filters and exporting again

## Import fails with "invalid CSV"
- Ensure the file is comma-separated
- Confirm UTF-8 encoding
- Make sure header row exists

## "Fatal error" during import
- Increase PHP memory limit
- Reduce batch size
- Disable conflicting plugins temporarily
- Check WooCommerce logs for related stack traces

## How do I find logs?
- WooCommerce → Status → Logs
- Or check your server error log

## How do I speed up export?
- Use smaller date ranges
- Export in batches
- Avoid expensive filters (like product-level filters) if not needed

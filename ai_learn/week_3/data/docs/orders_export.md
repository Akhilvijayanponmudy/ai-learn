# Orders Export - Filters & Fields

## Filters supported
- Date range (created date)
- Order status (processing, completed, refunded, etc.)
- Payment method
- Customer email
- Product / SKU filters (if enabled)
- Order total min/max (optional)

## Output columns
Common columns include:
- Order ID
- Order date
- Billing name, email
- Shipping address
- Line items (may be flattened or expanded based on settings)
- Taxes and discounts
- Payment method
- Order status

## Performance notes
Large exports can be slow.
Recommended settings:
- Reduce date range
- Export in batches
- Increase PHP memory limit
- Increase max execution time

## FAQ
### Why are line items not in separate rows?
Some exports generate one row per order. Line items may be included as a combined field.
If you need one row per line item, enable line-item mode (if available in your version).

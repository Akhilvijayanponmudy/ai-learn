# Users/Customers Import - Mapping & Meta

## Expected columns
- user_email (required)
- first_name
- last_name
- role (optional)
- billing_* fields (optional)
- shipping_* fields (optional)

## Updating existing users
If a user_email already exists:
- The importer can update user meta depending on settings.
- You can choose "Skip existing" or "Update existing".

## User meta formatting
For complex fields (like memberships):
- Use a delimiter (comma-separated) in a single column
- Example: memberships = "Gold,Silver"

## Common issues
### Duplicate email error
- The email already exists
- Decide whether to update existing users or skip

### Role not applied
- Ensure the role is a valid WordPress role slug
- Some hosting setups restrict role changes

## Tips
- Always test with 5–10 users first.
- Use UTF-8 CSV encoding.

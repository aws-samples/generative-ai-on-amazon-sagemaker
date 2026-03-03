# Retail Database Schema Overview

The **retail** database in AWS Glue contains **2 tables**: `orders` and `returns`. Here's a detailed description of each:

---

## Table 1: **orders**

### Schema Description:
The `orders` table is a comprehensive order and product transaction table containing 24 columns that track sales data across multiple dimensions.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| row_id | bigint | Unique row identifier |
| order_id | string | Unique order identifier |
| order_date | timestamp | Date when the order was placed |
| ship_date | timestamp | Date when the order was shipped |
| ship_mode | string | Shipping method (e.g., Same Day, Second Class, First Class) |
| customer_id | string | Unique customer identifier |
| customer_name | string | Name of the customer |
| segment | string | Customer segment (Consumer, Corporate, Home Office) |
| city | string | City of delivery |
| state | string | State/Province of delivery |
| country | string | Country of delivery |
| postal_code | double | Postal code of delivery |
| market | string | Market region (US, APAC, EU, LATAM, Africa) |
| region | string | Specific region within market |
| product_id | string | Unique product identifier |
| category | string | Product category (Technology, Furniture, Office Supplies) |
| sub-category | string | Product sub-category (Accessories, Chairs, Phones, etc.) |
| product_name | string | Name of the product |
| sales | double | Sales amount |
| quantity | bigint | Quantity ordered |
| discount | double | Discount percentage applied |
| profit | double | Profit amount |
| shipping_cost | double | Cost of shipping |
| order_priority | string | Priority level (Critical, High, Medium, Low) |

### Sample Data (5 rows):

| row_id | order_id | ship_mode | customer_name | city | state | country | market | category | sub-category | product_name | sales | quantity | discount | profit | order_priority |
|--------|----------|-----------|--------------|------|-------|---------|--------|----------|--------------|--------------|-------|----------|----------|--------|-----------------|
| 32298 | CA-2012-124891 | Same Day | Rick Hansen | New York City | New York | United States | US | Technology | Accessories | Plantronics CS510 - Over-the-Head monaural Wireless Headset System | 2309.65 | 7 | 0.0 | 762.18 | Critical |
| 26341 | IN-2013-77878 | Second Class | Justin Ritter | Wollongong | New South Wales | Australia | APAC | Furniture | Chairs | Novimex Executive Leather Armchair | 3709.00 | 9 | 0.1 | -288.77 | (N/A) |
| 25330 | IN-2013-71249 | First Class | Craig Reiter | Brisbane | Queensland | Australia | APAC | Technology | Phones | Nokia Smart Phone | 5175.00 | 9 | 0.1 | 919.97 | (N/A) |
| 13524 | ES-2013-1579342 | First Class | Katherine Murray | Berlin | Berlin | Germany | EU | Technology | Phones | Motorola Smart Phone | 2892.00 | 5 | 0.1 | -96.54 | (N/A) |
| 47221 | SG-2013-4320 | Same Day | Rick Hansen | Dakar | Dakar | Senegal | Africa | Technology | Copiers | Sharp Wireless Fax | 2832.00 | 8 | 0.0 | 311.52 | (N/A) |

---

## Table 2: **returns**

### Schema Description:
The `returns` table is a simpler table with 3 columns tracking product returns across different markets.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| returned | string | Whether the product was returned (Yes/No) |
| order_id | string | Unique order identifier (links to orders table) |
| market | string | Market region where the return occurred |

### Sample Data (5 rows):

| returned | order_id | market |
|----------|----------|--------|
| Yes | MX-2013-168137 | LATAM |
| Yes | US-2011-165316 | LATAM |
| Yes | ES-2013-1525878 | EU |
| Yes | CA-2013-118311 | United States |
| Yes | ES-2011-1276768 | EU |

---

## Usage Guidelines

- When writing a SQL query, always prefix it with the database name. For example, `retail.orders` or `retail.returns`. This is required for Amazon Athena compatibility.
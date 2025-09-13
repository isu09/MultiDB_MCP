Flow of create_table_tool

User gives input → A file path (like data.csv) and a table name (my_table).

Step 1: Read file schema

The file (CSV, Excel, JSON) is read.

From that file, we extract column names and their data types (e.g., user: int, user_name: string).

Step 2: Ask LLM to generate SQL

We send the schema and table name to the LLM (OpenAI).

The LLM generates a CREATE TABLE SQL query in MySQL syntax.

Step 3: Clean the query

The response might come with formatting (like inside ```sql blocks).

We extract only the SQL part to use directly.

Step 4: Execute SQL on MySQL

The SQL query is run against the MySQL database.

This creates the table.

Step 5: Return success message

If table creation is successful → return "Table created successfully!".

If not, errors will be raised (e.g., wrong SQL, DB connection issue).
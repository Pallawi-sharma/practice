---
title: SQL Basics
description: Learnings
owners: Pallawi
authors: Pallawi
categories: Learnings, references
tags: Learnings
---
 
# SQL Basics

## What is SQL?
* SQL (Structured Query Language) is used to:
  * Store data
  * Retrieve data
  * Update data
  * Delete data
 from a database.

* SQL works with relational databases like:
  * MySQL
  * PostgreSQL
  * SQL Server
  * Oracle
  * SQLite 

## What is a Database?
* A database is a place where data is stored.
  
## What is a Table?
* A table stores data in rows and columns.
* Example: **employees** table:
  
  | id | name  | age | department | salary |
  | -- | ----- | --- | ---------- | ------ |
  | 1  | Rahul | 25  | IT         | 50000  |
  | 2  | Anita | 28  | HR         | 45000  |

* Row → one record
* Column → one field

## Basic SQL Command
**SELECT Statement**
* Used to read data from a table.
  ```sql 
  SELECT * FROM employees;
  ```
* means all columns
  
**Select Specific Columns**
  ```sql
  SELECT name, salary FROM employees;
  ```
**WHERE Clause (Filtering Data)**
  ```sql
  SELECT * FROM employees
  WHERE department = 'IT';
  ```
**Using Conditions**
  ```sql
  SELECT * FROM employees
  WHERE salary > 45000;
  ```
**AND / OR**
  ```sql
  SELECT * FROM employees
  WHERE department = 'IT' AND salary > 40000;
  ```
**ORDER BY (Sorting Data)**
  * Used to sort data in ascending (ASC) or descending (DESC) order.
  * Example: Sort by age (youngest first)
  ```sql
  SELECT * FROM students
  ORDER BY age;
   ```
  * (Default is ASC)
  
**Oldest first**
  ```sql
  SELECT * FROM students
  ORDER BY age DESC;
  ```
**Sort by name alphabetically**
  ```sql
  SELECT * FROM students
  ORDER BY name ASC;
  ```
**LIMIT (Restrict Number of Rows)**
  * Used to limit the output rows.
  * Get first 5 students
  ```sql
  SELECT * FROM students
  LIMIT 5;
  ```
**Top 3 oldest students**
  ```sql
  SELECT * FROM students
  ORDER BY age DESC
  LIMIT 3;
  ```
**ORDER BY + WHERE + LIMIT (Very Important Combo)**
  * Get top 2 highest paid IT employees
  
  ```sql
  SELECT * FROM employees
  WHERE department = 'IT'
  ORDER BY salary DESC
  LIMIT 2;
  ```

**SQL Operators (IN, BETWEEN, LIKE)**
  * **IN Operator** - Used when you want to match multiple values.
  * **Instead of:** 
  ```sql 
  WHERE department = 'IT' OR department = 'HR'
  ```
  * **Use:**
  ```sql
   SELECT * FROM employees
   WHERE department IN ('IT', 'HR');
  ```
  * **BETWEEN Operator** - Used for a range of values.
  * **Example:**
  ```sql
  SELECT * FROM employees
  WHERE salary BETWEEN 40000 AND 60000;
  ```
  * Includes both 40000 & 60000
  * **LIKE Operator (Pattern Matching)** - Used with text.
  
    | Pattern | Meaning                  |
    | ------- | ------------------------ |
    | `%`     | Any number of characters |
    | `_`     | Exactly one character    |

  * Names starting with 'A'
  ```sql
    SELECT * FROM students
    WHERE name LIKE 'A%';
  ```
  * Names ending with 'n'
  ```sql
    SELECT * FROM students
    WHERE name LIKE '%n';
  ```
  * Names with 'a' in middle
  ```sql
    SELECT * FROM students
      WHERE name LIKE '%a%';
  ```




   
 




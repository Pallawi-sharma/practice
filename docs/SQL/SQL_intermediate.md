---
title: SQL Intermediate
description: Learnings
owners: Pallawi
authors: Pallawi
categories: Learnings, references
tags: Learnings
---
 
# SQL Intermediate

## Aggregate Functions

* Aggregate functions summarize data.

**COUNT()**
* Counts number of rows.
  
  ```sql 
  SELECT COUNT(*) FROM students;
  ```
* Count students from Science:
  ```sql
  SELECT COUNT(*) FROM students
  WHERE department = 'Science';
  ```

**SUM()**
* Adds values.
  
  ```sql
  SELECT SUM(salary) FROM employees;
  ```

**AVG()**
* Finds average.
  
  ```sql
  SELECT AVG(age) FROM students;
  ```

**MIN() & MAX()**

  ```sql
  SELECT MIN(age) FROM students;
  SELECT MAX(age) FROM students;
  ```

## GROUP BY

* GROUP BY is used when you want to group rows and apply aggregate functions.
* Example Table: students
    | id | name  | age | department |
    | -- | ----- | --- | ---------- |
    | 1  | Asha  | 20  | IT         |
    | 2  | Ravi  | 22  | IT         |
    | 3  | Neha  | 21  | Science    |
    | 4  | Aman  | 23  | Science    |
    | 5  | Kiran | 24  | HR         |

* **Count students in each department**
  ```sql
  SELECT department, COUNT(*) 
  FROM students
  GROUP BY department;
  ```
* **Rule:**
  * Every column in SELECT must be either
    * inside an aggregate function, or
    * included in GROUP BY
* **Average age per department**
  ```sql
  SELECT department, AVG(age)
  FROM students
  GROUP BY department;
  ```
* **GROUP BY + WHERE**
  ```sql
  SELECT department, COUNT(*)
  FROM students
  WHERE age > 20
  GROUP BY department;
  ```
* **HAVING (Very Important)**
  * HAVING is like WHERE but works after GROUP BY.
* This is wrong:
  * WHERE COUNT(*) > 2
* Correct:
  * HAVING COUNT(*) > 2
* Example:
  ```sql
  SELECT department, COUNT(*)
  FROM students
  GROUP BY department
  HAVING COUNT(*) > 2;
  ```

## SQL JOINS (CORE CONCEPT)

* In real databases, data is stored in multiple tables.
* JOINS help us combine them.
* Example Tables
* **students**
    | student_id | name | department_id |
    | ---------- | ---- | ------------- |
    | 1          | Asha | 101           |
    | 2          | Ravi | 102           |
    | 3          | Neha | 101           |

* **departments**
    | department_id | department_name |
    | ------------- | --------------- |
    | 101           | IT              |
    | 102           | Science         |
    | 103           | HR              |

**INNER JOIN**
* Returns only matching records from both tables.
  ```sql
  SELECT s.name, d.department_name
  FROM students s
  INNER JOIN departments d
  ON s.department_id = d.department_id;
  ```
* Students without a matching department are excluded.

**LEFT JOIN**
* Returns all records from left table + matching from right.
  ```sql
  SELECT s.name, d.department_name
  FROM students s
  LEFT JOIN departments d
  ON s.department_id = d.department_id;
  ```
* If no match → NULL values.

**RIGHT JOIN**
* Returns all records from right table + matching from left.
  ```sql
  SELECT s.name, d.department_name
  FROM students s
  RIGHT JOIN departments d
  ON s.department_id = d.department_id;
  ```

**FULL JOIN**
* Returns all records from both tables.
  ```sql
  SELECT s.name, d.department_name
  FROM students s
  FULL JOIN departments d
  ON s.department_id = d.department_id;
  ```

* Very Important Concept
  
    | Join Type | What it returns          |
    | --------- | ------------------------ |
    | INNER     | Matching rows only       |
    | LEFT      | All left + matched right |
    | RIGHT     | All right + matched left |
    | FULL      | Everything               |

## JOIN + WHERE + GROUP BY (Real-World Queries)

**Count students in each department**
  ```sql
  SELECT d.department_name, COUNT(s.student_id)
  FROM departments d
  LEFT JOIN students s
  ON s.department_id = d.department_id
  GROUP BY d.department_name;
  ```
**Departments with more than 2 students**
```sql
SELECT d.department_name, COUNT(s.student_id)
FROM departments d
JOIN students s
ON s.department_id = d.department_id
GROUP BY d.department_name
HAVING COUNT(s.student_id) > 2;

```





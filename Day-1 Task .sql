CREATE DATABASE CompanyDB;
USE CompanyDB;
CREATE TABLE IF NOT EXISTS Departments (
    DepartmentID INT PRIMARY KEY,
    DepartmentName VARCHAR(50) NOT NULL
);
CREATE TABLE IF NOT EXISTS Employees (
    EmployeeID INT PRIMARY KEY,
    FirstName VARCHAR(50) NOT NULL,
    LastName VARCHAR(50) NOT NULL,
    Salary DECIMAL(10,2),
    DepartmentID INT,
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
);

SELECT * FROM Departments;
INSERT INTO Departments (DepartmentID, DepartmentName) VALUES
(1, 'Human Resources'),
(2, 'Finance'),
(3, 'Engineering');

INSERT INTO Employees (EmployeeID, FirstName, LastName, Salary, DepartmentID) VALUES
(101, 'Alice', 'Smith', 60000, 1),
(102, 'Bob', 'Johnson', 75000, 2),
(103, 'Charlie', 'Lee', 50000, 3),
(104, 'Diana', 'Wright', 90000, 3),
(105, 'Eva', 'Martinez', 55000, 1),
(106, 'Frank', 'Taylor', 82000, 2);
SELECT FirstName, LastName, Salary
FROM Employees;
SELECT FirstName, LastName, Salary
FROM Employees
WHERE Salary > 60000;
SELECT FirstName, LastName, Salary
FROM Employees
ORDER BY Salary DESC;
SELECT d.DepartmentName, AVG(e.Salary) AS AvgSalary
FROM Employees e
JOIN Departments d ON e.DepartmentID = d.DepartmentID
GROUP BY d.DepartmentName
HAVING AVG(e.Salary) > 60000;

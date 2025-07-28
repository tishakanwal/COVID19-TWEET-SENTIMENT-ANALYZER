USE CompanyDB;
SELECT 
    e.EmployeeID, 
    e.FirstName, 
    e.LastName, 
    e.Salary, 
    d.DepartmentName
FROM 
    Employees e
JOIN 
    Departments d ON e.DepartmentID = d.DepartmentID;

SELECT 
    FirstName, 
    LastName, 
    Salary
FROM 
    Employees
WHERE 
    Salary > (
        SELECT AVG(Salary) FROM Employees
    );
    
    SELECT 
    d.DepartmentName, 
    SUM(e.Salary) AS TotalSalary
FROM 
    Employees e
JOIN 
    Departments d ON e.DepartmentID = d.DepartmentID
GROUP BY 
    d.DepartmentName;


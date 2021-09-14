select FirstName, LastName, City, State
from Person left join Address
on Person.PersonId = Address.PersonId

回忆一下 limit 的用法

limit N     # 返回 N 条记录
offset M    # 跳过 M 条记录，M 默认为 0
limit M,N   # 相当于 limit N offset M，从第 M 条记录开始，返回 N 条记录
将 Salary 去重后降序排列，再返回第二条记录可得第二大的值
也许只有一个 Salary 值，将返回 null

select (
	select DISTINCT Salary
	from Employee
	order by Salary DESC
	limit 1,1) 
as SecondHighestSalary;


CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  set n = N-1;
  RETURN (
      # Write your MySQL query statement below.
      select ifnull(
      (
      select distinct Salary getNthHighestSalary
      from Employee
      order by Salary desc
      limit n, 1
      ), null)
  );
END


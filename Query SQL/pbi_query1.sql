select
	"Marital Status",
	round(avg(age), 2) "AVG Age"
from
	customers c
where
	"Marital Status" not in ('')
group by 1
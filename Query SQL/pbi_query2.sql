select
	case
		when gender = 0 then 'Wanita'
		when gender = 1 then 'Pria'
	end gender,
	round(avg(age),
	2) "AVG Age"
from
	customers c
group by
	1
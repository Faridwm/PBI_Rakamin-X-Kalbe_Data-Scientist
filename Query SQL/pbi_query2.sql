select
	case
		when gender = 0 then 'Wanita'
		when gender = 1 then 'Pria'
	end gender,
	concat(round(avg(age)), ' Tahun') "AVG Age"
from
	customers c
group by
	1
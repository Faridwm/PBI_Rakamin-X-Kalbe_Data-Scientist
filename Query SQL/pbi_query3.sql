select
	s.storename,
	sum(t.qty) "Sum QTY"
from
	stores s
join transactions t on
	s.storeid = t.storeid
group by 1
order by 2 desc 
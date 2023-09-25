select
	p."Product Name",
	sum(t.totalamount) as "total amount"
from
	products p
join transactions t on
	p.productid = t.productid
group by 1
order by 2 desc
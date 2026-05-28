SELECT * FROM futures.ticks_raw
WHERE symbol = %(symbol)s AND date BETWEEN %(start_date)s AND %(end_date)s
ORDER BY date, time, sequence;

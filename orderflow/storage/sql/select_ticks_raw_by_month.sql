SELECT * FROM futures.ticks_raw
WHERE symbol = %(symbol)s AND month_key = %(month_key)s
ORDER BY date, time, sequence;

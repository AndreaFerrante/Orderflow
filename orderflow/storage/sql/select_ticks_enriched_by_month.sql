SELECT * FROM futures.ticks_enriched
WHERE symbol = %(symbol)s AND month_key = %(month_key)s
ORDER BY "Date", "Time", "Sequence";

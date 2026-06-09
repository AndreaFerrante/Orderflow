SELECT * FROM futures.ticks_enriched
WHERE symbol = %(symbol)s AND "Date" BETWEEN %(start_date)s AND %(end_date)s
ORDER BY "Date", "Time", "Sequence";

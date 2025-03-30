import os
import ast
import time
import polars as pl

def read_and_clean_trades(trade_list_path:str, trade_list_name:str):

    trades = pl.read_csv(
        os.path.join(trade_list_path, trade_list_name),
        separator             = "\t",
        encoding              = "utf-8",
        has_header            = True,
        ignore_errors         = True,
        truncate_ragged_lines = True
    )

    trades = trades.with_columns(pl.col('Entry DateTime').str.replace(' BP', ''))
    trades = trades.with_columns(pl.col('Exit DateTime').str.replace(' EP', ''))
    trades = trades.with_columns(pl.col('Entry DateTime').str.split(by='  '))
    trades = trades.with_columns(pl.col('Exit DateTime').str.split(by='  '))

    entry_date = list()
    entry_time = list()

    for entry_times in trades['Entry DateTime']:
        entry_date.append(entry_times[0])
        entry_time.append(entry_times[1])

    trades = trades.with_columns(pl.Series(name='Entry Date', values=entry_date))
    trades = trades.with_columns(pl.Series(name='Entry Time', values=entry_time))

    exit_date = list()
    exit_time = list()

    for exit_times in trades['Exit DateTime']:
        exit_date.append(exit_times[0])
        exit_time.append(exit_times[1])

    trades = trades.with_columns(pl.Series(name='Exit Date', values=exit_date))
    trades = trades.with_columns(pl.Series(name='Exit Time', values=exit_time))
    trades = trades.with_columns(pl.concat_str([pl.col("Exit Date"),
                                                pl.col("Exit Time")],
                                                separator=" ")
                                                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False)
                                                .alias("ExitDateTime"))
    trades = trades.with_columns(pl.concat_str([pl.col("Entry Date"),
                                                pl.col("Entry Time")],
                                                separator=" ")
                                                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False)
                                                .alias("EntryDateTime"))
    trades = trades.with_columns(Index = pl.arange(1, len(trades) + 1))
    trades = trades.sort(["EntryDateTime"], descending=False)

    return trades

def match_trades(trades, data, trades_col='EntryDateTime', data_col='DateTime', unique=True) -> pl.DataFrame:

    try:

        counter       = 0
        selected_rows = list()
        all_rows      = len(trades)
        start_time    = time.time()

        for trade in trades.iter_rows(named=True):

            entry_time = trade[trades_col]

            for tick in data.iter_rows(named=True):

                if entry_time >= tick[data_col]:
                    selected_rows.append(tick)
                    print(f"Entry time: {entry_time}, Data DateTime: {tick[data_col]}")
                    break

            counter += 1
            #print(f"Processed till {round(counter / all_rows, 3) * 100} percent ...")

        print(f"It took {round(time.time() - start_time, 3)} seconds.")

        selected_rows = pl.DataFrame( selected_rows )
        if unique:
            selected_rows = selected_rows.unique(subset="Index", keep='first')

        return pl.DataFrame( selected_rows )

    except Exception as ex:
        raise Exception(f"While matching th trades, this issue occured: {ex}")

def clean_notes(notes: pl.Series, indexes: pl.Series, target:pl.Series):

    try:

        data, lines, labels = list(), list(), list()

        for note, index, label in zip(notes, indexes, target):

            single_notes = note.split(';')
            header       = list()
            values       = list()

            for single_note in single_notes:

                if single_note != ' ':

                    single_note = str(single_note).strip()
                    col_value   = str(single_note.split(':')[0]).strip()

                    try:
                        param_value = ast.literal_eval(str(single_note.split(':')[1]).strip())
                    except Exception as ex:
                        param_value = str(single_note.split(':')[1]).strip()

                    # Fill in the header and its associated value ...
                    header.append(col_value)
                    values.append(param_value)

            lines.append(index)
            labels.append(label)
            data.append(pl.DataFrame({x: y for x, y in zip(header, values)}))

        all_data   = pl.concat(data)
        all_labels = pl.DataFrame({'Labels': labels})
        all_labels = all_labels.with_columns(pl.when(pl.col('Labels') > 0).then(1).otherwise(0).alias('Target'))
        all_lines  = pl.DataFrame({'Index': lines})

        return pl.concat([all_data, all_lines, all_labels], how='horizontal')

    except Exception as ex:
        raise Exception(f"While reading the data we saw this exception: {ex}")

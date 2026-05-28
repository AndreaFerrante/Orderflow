CREATE TABLE futures.ticks_raw (
    symbol          VARCHAR(10)     NOT NULL,
    month_key       VARCHAR(6)      NOT NULL,
    date            DATE            NOT NULL,
    time            TIME(3)         NOT NULL,
    sequence        BIGINT          NOT NULL,
    depth_sequence  BIGINT,
    price           DOUBLE PRECISION,
    volume          INTEGER,
    trade_type      SMALLINT,
    ask_price       DOUBLE PRECISION,
    bid_price       DOUBLE PRECISION,
    ask_size        INTEGER,
    bid_size        INTEGER,
    total_ask_depth INTEGER,
    total_bid_depth INTEGER,
    ask_dom_price   DOUBLE PRECISION,
    bid_dom_price   DOUBLE PRECISION,
    ask_dom_0       INTEGER,  bid_dom_0  INTEGER,
    ask_dom_1       INTEGER,  bid_dom_1  INTEGER,
    ask_dom_2       INTEGER,  bid_dom_2  INTEGER,
    ask_dom_3       INTEGER,  bid_dom_3  INTEGER,
    ask_dom_4       INTEGER,  bid_dom_4  INTEGER,
    ask_dom_5       INTEGER,  bid_dom_5  INTEGER,
    ask_dom_6       INTEGER,  bid_dom_6  INTEGER,
    ask_dom_7       INTEGER,  bid_dom_7  INTEGER,
    ask_dom_8       INTEGER,  bid_dom_8  INTEGER,
    ask_dom_9       INTEGER,  bid_dom_9  INTEGER,
    ask_dom_10      INTEGER,  bid_dom_10 INTEGER,
    ask_dom_11      INTEGER,  bid_dom_11 INTEGER,
    ask_dom_12      INTEGER,  bid_dom_12 INTEGER,
    ask_dom_13      INTEGER,  bid_dom_13 INTEGER,
    ask_dom_14      INTEGER,  bid_dom_14 INTEGER,
    ask_dom_15      INTEGER,  bid_dom_15 INTEGER,
    ask_dom_16      INTEGER,  bid_dom_16 INTEGER,
    ask_dom_17      INTEGER,  bid_dom_17 INTEGER,
    ask_dom_18      INTEGER,  bid_dom_18 INTEGER,
    ask_dom_19      INTEGER,  bid_dom_19 INTEGER,
    ask_dom_20      INTEGER,  bid_dom_20 INTEGER,
    ask_dom_21      INTEGER,  bid_dom_21 INTEGER,
    ask_dom_22      INTEGER,  bid_dom_22 INTEGER,
    ask_dom_23      INTEGER,  bid_dom_23 INTEGER,
    ask_dom_24      INTEGER,  bid_dom_24 INTEGER,
    ask_dom_25      INTEGER,  bid_dom_25 INTEGER,
    ask_dom_26      INTEGER,  bid_dom_26 INTEGER,
    ask_dom_27      INTEGER,  bid_dom_27 INTEGER,
    ask_dom_28      INTEGER,  bid_dom_28 INTEGER,
    ask_dom_29      INTEGER,  bid_dom_29 INTEGER,
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP       NOT NULL DEFAULT NOW()
) PARTITION BY LIST (symbol);

-- Primary key on staging
ALTER TABLE futures.ticks_raw
    ADD CONSTRAINT pk_ticks_raw
    PRIMARY KEY (symbol, month_key, date, time, sequence);

CREATE TABLE futures.ticks_raw_es PARTITION OF futures.ticks_raw
    FOR VALUES IN ('ES')
    PARTITION BY RANGE (month_key);

CREATE TABLE futures.ticks_raw_es_202501 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202501') TO ('202502');
CREATE TABLE futures.ticks_raw_es_202502 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202502') TO ('202503');
CREATE TABLE futures.ticks_raw_es_202503 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202503') TO ('202504');
CREATE TABLE futures.ticks_raw_es_202504 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202504') TO ('202505');
CREATE TABLE futures.ticks_raw_es_202505 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202505') TO ('202506');
CREATE TABLE futures.ticks_raw_es_202506 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202506') TO ('202507');
CREATE TABLE futures.ticks_raw_es_202507 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202507') TO ('202508');
CREATE TABLE futures.ticks_raw_es_202508 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202508') TO ('202509');
CREATE TABLE futures.ticks_raw_es_202509 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202509') TO ('202510');
CREATE TABLE futures.ticks_raw_es_202510 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202510') TO ('202511');
CREATE TABLE futures.ticks_raw_es_202511 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202511') TO ('202512');
CREATE TABLE futures.ticks_raw_es_202512 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202512') TO ('202601');
CREATE TABLE futures.ticks_raw_es_202601 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202601') TO ('202602');
CREATE TABLE futures.ticks_raw_es_202602 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202602') TO ('202603');
CREATE TABLE futures.ticks_raw_es_202603 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202603') TO ('202604');
CREATE TABLE futures.ticks_raw_es_202604 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202604') TO ('202605');
CREATE TABLE futures.ticks_raw_es_202605 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202605') TO ('202606');
CREATE TABLE futures.ticks_raw_es_202606 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202606') TO ('202607');
CREATE TABLE futures.ticks_raw_es_202607 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202607') TO ('202608');
CREATE TABLE futures.ticks_raw_es_202608 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202608') TO ('202609');
CREATE TABLE futures.ticks_raw_es_202609 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202609') TO ('202610');
CREATE TABLE futures.ticks_raw_es_202610 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202610') TO ('202611');
CREATE TABLE futures.ticks_raw_es_202611 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202611') TO ('202612');
CREATE TABLE futures.ticks_raw_es_202612 PARTITION OF futures.ticks_raw_es
    FOR VALUES FROM ('202612') TO ('202701');

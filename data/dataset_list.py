
TIME = 'time'
LOW = 'low'
HIGH = 'high'
OPEN = 'open'
CLOSE = 'close'
VOLUME = 'volume'
QUOTE_VOLUME = 'quoteVolume'
AVERAGE = 'average'
WEIGHTED_AVERAGE = 'weightedAverage'
NUM_TRADES = 'numTrades'
BUY_BASE_VOLUME = 'buyBaseVolume'
BUY_QUOTE_VOLUME = 'buyQuoteVolume'
NUM_BLOCKS = 'numBlocks'
AVERAGE_DIFFICULTY = 'avgDifficulty'
AVERAGE_TOTAL_DIFFICULTY = 'avgTotalDifficulty'
GAS_USED = 'gasUsed'
BLOCK_SIZE = 'blockSize'
NUM_TRANSACTIONS = 'numTransactions'
VALUE = 'value'
NUM_CONTRACTS = 'numContracts'
CONTRACT_GAS = 'contractGas'
CONTRACT_VALUE = 'contractValue'
IS_CARRIED = 'isCarried'

GDAX_CHART = 'gdax_chart'
POLO_CHART = 'polo_chart'
POLO_TRADE = 'polo_trade'
BINA_CHART = 'bina_chart'
ETHE_BLOCK = 'ethereum'

LABELS = {
    GDAX_CHART: [TIME,
                 LOW,
                 HIGH,
                 OPEN,
                 CLOSE,
                 VOLUME,
                 IS_CARRIED],
    POLO_CHART: [TIME,
                 LOW,
                 HIGH,
                 OPEN,
                 CLOSE,
                 VOLUME,
                 QUOTE_VOLUME,
                 WEIGHTED_AVERAGE,
                 IS_CARRIED],
    POLO_TRADE: [TIME,
                 NUM_TRADES,
                 VOLUME,
                 AVERAGE,
                 WEIGHTED_AVERAGE,
                 IS_CARRIED],
    BINA_CHART: [TIME,
                 LOW,
                 HIGH,
                 OPEN,
                 CLOSE,
                 VOLUME,
                 QUOTE_VOLUME,
                 NUM_TRADES,
                 BUY_BASE_VOLUME,
                 BUY_QUOTE_VOLUME,
                 IS_CARRIED],
    ETHE_BLOCK: [TIME,
                 NUM_BLOCKS,
                 AVERAGE_DIFFICULTY,
                 AVERAGE_TOTAL_DIFFICULTY,
                 GAS_USED,
                 BLOCK_SIZE,
                 NUM_TRANSACTIONS,
                 VALUE,
                 NUM_CONTRACTS,
                 CONTRACT_GAS,
                 CONTRACT_VALUE]
}


JAN2018 = {
    GDAX_CHART: ['ETH-USD', 'ETH-BTC', 'ETH-EUR', 'BTC-USD', 'BTC-EUR',
                 'BTC-GBP', 'LTC-USD', 'LTC-EUR', 'LTC-BTC', 'BCH-USD'],
    POLO_CHART: ['BTC_AMP', 'BTC_ARDR', 'BTC_BCH', 'BTC_BCN', 'BTC_BCY',
                 'BTC_BELA', 'BTC_BLK', 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS',
                 'BTC_BURST', 'BTC_CLAM', 'BTC_CVC', 'BTC_DASH', 'BTC_DCR',
                 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2', 'BTC_ETC', 'BTC_ETH',
                 'BTC_EXP', 'BTC_FCT', 'BTC_FLDC', 'BTC_FLO', 'BTC_GAME',
                 'BTC_GAS', 'BTC_GNO', 'BTC_GNT', 'BTC_GRC', 'BTC_HUC',
                 'BTC_LBC', 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NAV',
                 'BTC_NEOS', 'BTC_NMC', 'BTC_NXC', 'BTC_NXT', 'BTC_OMG',
                 'BTC_OMNI', 'BTC_PASC', 'BTC_PINK', 'BTC_POT', 'BTC_PPC',
                 'BTC_RADS', 'BTC_REP', 'BTC_RIC', 'BTC_SBD', 'BTC_SC',
                 'BTC_STEEM', 'BTC_STORJ', 'BTC_STR', 'BTC_STRAT', 'BTC_SYS',
                 'BTC_VIA', 'BTC_VRC', 'BTC_VTC', 'BTC_XBC', 'BTC_XCP',
                 'BTC_XEM', 'BTC_XMR', 'BTC_XPM', 'BTC_XRP', 'BTC_XVC',
                 'BTC_ZEC', 'BTC_ZRX', 'ETH_BCH', 'ETH_CVC', 'ETH_ETC',
                 'ETH_GAS', 'ETH_GNO', 'ETH_GNT', 'ETH_LSK', 'ETH_OMG',
                 'ETH_REP', 'ETH_STEEM', 'ETH_ZEC', 'ETH_ZRX', 'USDT_BCH',
                 'USDT_BTC', 'USDT_DASH', 'USDT_ETC', 'USDT_ETH', 'USDT_LTC',
                 'USDT_NXT', 'USDT_REP', 'USDT_STR', 'USDT_XMR', 'USDT_XRP',
                 'USDT_ZEC', 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH',
                 'XMR_LTC', 'XMR_MAID', 'XMR_NXT', 'XMR_ZEC'],
    POLO_TRADE: ['BTC_AMP', 'BTC_ARDR', 'BTC_BCH', 'BTC_BCN', 'BTC_BCY',
                 'BTC_BELA', 'BTC_BLK', 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS',
                 'BTC_BURST', 'BTC_CLAM', 'BTC_CVC', 'BTC_DASH', 'BTC_DCR',
                 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2', 'BTC_ETC', 'BTC_ETH',
                 'BTC_EXP', 'BTC_FCT', 'BTC_FLDC', 'BTC_FLO', 'BTC_GAME',
                 'BTC_GAS', 'BTC_GNO', 'BTC_GNT', 'BTC_GRC', 'BTC_HUC',
                 'BTC_LBC', 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NAV',
                 'BTC_NEOS', 'BTC_NMC', 'BTC_NXC', 'BTC_NXT', 'BTC_OMG',
                 'BTC_OMNI', 'BTC_PASC', 'BTC_PINK', 'BTC_POT', 'BTC_PPC',
                 'BTC_RADS', 'BTC_REP', 'BTC_RIC', 'BTC_SBD', 'BTC_SC',
                 'BTC_STEEM', 'BTC_STORJ', 'BTC_STR', 'BTC_STRAT', 'BTC_SYS',
                 'BTC_VIA', 'BTC_VRC', 'BTC_VTC', 'BTC_XBC', 'BTC_XCP',
                 'BTC_XEM', 'BTC_XMR', 'BTC_XPM', 'BTC_XRP', 'BTC_XVC',
                 'BTC_ZEC', 'BTC_ZRX', 'ETH_BCH', 'ETH_CVC', 'ETH_ETC',
                 'ETH_GAS', 'ETH_GNO', 'ETH_GNT', 'ETH_LSK', 'ETH_OMG',
                 'ETH_REP', 'ETH_STEEM', 'ETH_ZEC', 'ETH_ZRX', 'USDT_BCH',
                 'USDT_BTC', 'USDT_DASH', 'USDT_ETC', 'USDT_ETH', 'USDT_LTC',
                 'USDT_NXT', 'USDT_REP', 'USDT_STR', 'USDT_XMR', 'USDT_XRP',
                 'USDT_ZEC', 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH',
                 'XMR_LTC', 'XMR_MAID', 'XMR_NXT', 'XMR_ZEC'],
    BINA_CHART: ['ETHBTC', 'LTCBTC', 'BNBBTC', 'NEOBTC', 'QTUMETH', 'EOSETH',
                 'SNTETH', 'BNTETH', 'BCCBTC', 'GASBTC', 'BNBETH', 'BTCUSDT',
                 'ETHUSDT', 'HSRBTC', 'OAXETH', 'DNTETH', 'MCOETH', 'ICNETH',
                 'MCOBTC', 'WTCBTC', 'WTCETH', 'LRCBTC', 'LRCETH', 'QTUMBTC',
                 'YOYOBTC', 'OMGBTC', 'OMGETH', 'ZRXBTC', 'ZRXETH',
                 'STRATBTC', 'STRATETH', 'SNGLSBTC', 'SNGLSETH', 'BQXBTC',
                 'BQXETH', 'KNCBTC', 'KNCETH', 'FUNBTC', 'FUNETH', 'SNMBTC',
                 'SNMETH', 'NEOETH', 'IOTABTC', 'IOTAETH', 'LINKBTC',
                 'LINKETH', 'XVGBTC', 'XVGETH', 'CTRBTC', 'CTRETH', 'SALTBTC',
                 'SALTETH', 'MDABTC', 'MDAETH', 'MTLBTC', 'MTLETH', 'SUBBTC',
                 'SUBETH', 'EOSBTC', 'SNTBTC', 'ETCETH', 'ETCBTC', 'MTHBTC',
                 'MTHETH', 'ENGBTC', 'ENGETH', 'DNTBTC', 'ZECBTC', 'ZECETH',
                 'BNTBTC', 'ASTBTC', 'ASTETH', 'DASHBTC', 'DASHETH', 'OAXBTC',
                 'ICNBTC', 'BTGBTC', 'BTGETH', 'EVXBTC', 'EVXETH', 'REQBTC',
                 'REQETH', 'VIBBTC', 'VIBETH', 'HSRETH', 'TRXBTC', 'TRXETH',
                 'POWRBTC', 'POWRETH', 'ARKBTC', 'ARKETH', 'YOYOETH',
                 'XRPBTC', 'XRPETH', 'MODBTC', 'MODETH', 'ENJBTC', 'ENJETH',
                 'STORJBTC', 'STORJETH', 'BNBUSDT', 'VENBNB', 'YOYOBNB',
                 'POWRBNB', 'VENBTC', 'VENETH', 'KMDBTC', 'KMDETH', 'NULSBNB',
                 'RCNBTC', 'RCNETH', 'RCNBNB', 'NULSBTC', 'NULSETH', 'RDNBTC',
                 'RDNETH', 'RDNBNB', 'XMRBTC', 'XMRETH', 'DLTBNB', 'WTCBNB',
                 'DLTBTC', 'DLTETH', 'AMBBTC', 'AMBETH', 'AMBBNB', 'BCCETH',
                 'BCCUSDT', 'BCCBNB', 'BATBTC', 'BATETH', 'BATBNB', 'BCPTBTC',
                 'BCPTETH', 'BCPTBNB', 'ARNBTC', 'ARNETH', 'GVTBTC', 'GVTETH',
                 'CDTBTC', 'CDTETH', 'GXSBTC', 'GXSETH', 'NEOUSDT', 'NEOBNB',
                 'POEBTC', 'POEETH', 'QSPBTC', 'QSPETH', 'QSPBNB', 'BTSBTC',
                 'BTSETH', 'BTSBNB', 'XZCBTC', 'XZCETH', 'XZCBNB', 'LSKBTC',
                 'LSKETH', 'LSKBNB', 'TNTBTC', 'TNTETH', 'FUELBTC', 'FUELETH',
                 'MANABTC', 'MANAETH', 'BCDBTC', 'BCDETH', 'DGDBTC', 'DGDETH',
                 'IOTABNB', 'ADXBTC', 'ADXETH', 'ADXBNB', 'ADABTC', 'ADAETH',
                 'PPTBTC', 'PPTETH', 'CMTBTC', 'CMTETH', 'CMTBNB', 'XLMBTC',
                 'XLMETH', 'XLMBNB', 'CNDBTC', 'CNDETH', 'CNDBNB', 'LENDBTC',
                 'LENDETH', 'WABIBTC', 'WABIETH', 'WABIBNB', 'LTCETH',
                 'LTCUSDT', 'LTCBNB', 'TNBBTC', 'TNBETH', 'WAVESBTC',
                 'WAVESETH', 'WAVESBNB', 'GTOBTC', 'GTOETH', 'GTOBNB',
                 'ICXBTC', 'ICXETH', 'ICXBNB', 'OSTBTC', 'OSTETH', 'OSTBNB',
                 'ELFBTC', 'ELFETH', 'AIONBTC', 'AIONETH', 'AIONBNB',
                 'NEBLBTC', 'NEBLETH', 'NEBLBNB', 'BRDBTC', 'BRDETH',
                 'BRDBNB', 'MCOBNB', 'EDOBTC', 'EDOETH', 'WINGSBTC',
                 'WINGSETH', 'NAVBTC', 'NAVETH', 'NAVBNB', 'LUNBTC', 'LUNETH',
                 'TRIGBTC', 'TRIGETH', 'TRIGBNB'],
    ETHE_BLOCK: [ETHE_BLOCK]
}


MAR2017 = {
    GDAX_CHART: ['ETH-USD', 'ETH-BTC', 'BTC-USD', 'BTC-EUR', 'BTC-GBP',
                 'LTC-USD'],
    POLO_CHART: ['BTC_AMP', 'BTC_ARDR', 'BTC_BCN', 'BTC_BCY', 'BTC_BELA',
                 'BTC_BLK', 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS', 'BTC_BURST',
                 'BTC_CLAM', 'BTC_DASH', 'BTC_DCR', 'BTC_DGB', 'BTC_DOGE',
                 'BTC_ETC', 'BTC_ETH', 'BTC_EXP', 'BTC_FCT', 'BTC_FLDC',
                 'BTC_FLO', 'BTC_GAME', 'BTC_GNT', 'BTC_GRC', 'BTC_HUC',
                 'BTC_LBC', 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NAV',
                 'BTC_NEOS', 'BTC_NMC', 'BTC_NXC', 'BTC_NXT', 'BTC_OMNI',
                 'BTC_PASC', 'BTC_PINK', 'BTC_POT', 'BTC_PPC', 'BTC_RADS',
                 'BTC_REP', 'BTC_RIC', 'BTC_SBD', 'BTC_SC', 'BTC_STEEM',
                 'BTC_STR', 'BTC_STRAT', 'BTC_SYS', 'BTC_VIA', 'BTC_VRC',
                 'BTC_VTC', 'BTC_XBC', 'BTC_XCP', 'BTC_XEM', 'BTC_XMR',
                 'BTC_XPM', 'BTC_XRP', 'BTC_XVC', 'BTC_ZEC', 'ETH_ETC',
                 'ETH_GNT', 'ETH_LSK', 'ETH_REP', 'ETH_STEEM', 'ETH_ZEC',
                 'USDT_BTC', 'USDT_DASH', 'USDT_ETC', 'USDT_ETH', 'USDT_LTC',
                 'USDT_NXT', 'USDT_REP', 'USDT_STR', 'USDT_XMR', 'USDT_XRP',
                 'USDT_ZEC', 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH',
                 'XMR_LTC', 'XMR_MAID', 'XMR_NXT', 'XMR_ZEC'],
    POLO_TRADE: ['BTC_AMP', 'BTC_ARDR', 'BTC_BCN', 'BTC_BCY', 'BTC_BELA',
                 'BTC_BLK', 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS', 'BTC_BURST',
                 'BTC_CLAM', 'BTC_DASH', 'BTC_DCR', 'BTC_DGB', 'BTC_DOGE',
                 'BTC_ETC', 'BTC_ETH', 'BTC_EXP', 'BTC_FCT', 'BTC_FLDC',
                 'BTC_FLO', 'BTC_GAME', 'BTC_GNT', 'BTC_GRC', 'BTC_HUC',
                 'BTC_LBC', 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NAV',
                 'BTC_NEOS', 'BTC_NMC', 'BTC_NXC', 'BTC_NXT', 'BTC_OMNI',
                 'BTC_PASC', 'BTC_PINK', 'BTC_POT', 'BTC_PPC', 'BTC_RADS',
                 'BTC_REP', 'BTC_RIC', 'BTC_SBD', 'BTC_SC', 'BTC_STEEM',
                 'BTC_STR', 'BTC_STRAT', 'BTC_SYS', 'BTC_VIA', 'BTC_VRC',
                 'BTC_VTC', 'BTC_XBC', 'BTC_XCP', 'BTC_XEM', 'BTC_XMR',
                 'BTC_XPM', 'BTC_XRP', 'BTC_XVC', 'BTC_ZEC', 'ETH_ETC',
                 'ETH_GNT', 'ETH_LSK', 'ETH_REP', 'ETH_STEEM', 'ETH_ZEC',
                 'USDT_BTC', 'USDT_DASH', 'USDT_ETC', 'USDT_ETH', 'USDT_LTC',
                 'USDT_NXT', 'USDT_REP', 'USDT_STR', 'USDT_XMR', 'USDT_XRP',
                 'USDT_ZEC', 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH',
                 'XMR_LTC', 'XMR_MAID', 'XMR_NXT', 'XMR_ZEC'],
    ETHE_BLOCK: [ETHE_BLOCK],
    BINA_CHART: []
}


AUG2016 = {
    GDAX_CHART: ['ETH-USD', 'ETH-BTC', 'BTC-USD', 'BTC-EUR'],
    POLO_CHART: ['BTC_AMP', 'BTC_BCN', 'BTC_BCY', 'BTC_BELA', 'BTC_BLK',
                 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS', 'BTC_BURST', 'BTC_CLAM',
                 'BTC_DASH', 'BTC_DCR', 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2',
                 'BTC_ETC', 'BTC_ETH', 'BTC_EXP', 'BTC_FCT', 'BTC_FLDC',
                 'BTC_FLO', 'BTC_GAME', 'BTC_GRC', 'BTC_HUC', 'BTC_LBC',
                 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NAV', 'BTC_NEOS',
                 'BTC_NMC', 'BTC_NXT', 'BTC_OMNI', 'BTC_PINK', 'BTC_POT',
                 'BTC_PPC', 'BTC_RADS', 'BTC_RIC', 'BTC_SBD', 'BTC_SC',
                 'BTC_STEEM', 'BTC_STR', 'BTC_SYS', 'BTC_VIA', 'BTC_VRC',
                 'BTC_VTC', 'BTC_XBC', 'BTC_XCP', 'BTC_XEM', 'BTC_XMR',
                 'BTC_XPM', 'BTC_XRP', 'BTC_XVC', 'ETH_ETC', 'ETH_LSK',
                 'ETH_STEEM', 'USDT_BTC', 'USDT_DASH', 'USDT_ETC', 'USDT_ETH',
                 'USDT_LTC', 'USDT_NXT', 'USDT_STR', 'USDT_XMR', 'USDT_XRP',
                 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH', 'XMR_LTC',
                 'XMR_MAID', 'XMR_NXT'],
    POLO_TRADE: ['BTC_AMP', 'BTC_BCN', 'BTC_BCY', 'BTC_BELA', 'BTC_BLK',
                 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS', 'BTC_BURST', 'BTC_CLAM',
                 'BTC_DASH', 'BTC_DCR', 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2',
                 'BTC_ETC', 'BTC_ETH', 'BTC_EXP', 'BTC_FCT', 'BTC_FLDC',
                 'BTC_FLO', 'BTC_GAME', 'BTC_GRC', 'BTC_HUC', 'BTC_LBC',
                 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NAV', 'BTC_NEOS',
                 'BTC_NMC', 'BTC_NXT', 'BTC_OMNI', 'BTC_PINK', 'BTC_POT',
                 'BTC_PPC', 'BTC_RADS', 'BTC_RIC', 'BTC_SBD', 'BTC_SC',
                 'BTC_STEEM', 'BTC_STR', 'BTC_SYS', 'BTC_VIA', 'BTC_VRC',
                 'BTC_VTC', 'BTC_XBC', 'BTC_XCP', 'BTC_XEM', 'BTC_XMR',
                 'BTC_XPM', 'BTC_XRP', 'BTC_XVC', 'ETH_ETC', 'ETH_LSK',
                 'ETH_STEEM', 'USDT_BTC', 'USDT_DASH', 'USDT_ETC', 'USDT_ETH',
                 'USDT_LTC', 'USDT_NXT', 'USDT_STR', 'USDT_XMR', 'USDT_XRP',
                 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH', 'XMR_LTC',
                 'XMR_MAID', 'XMR_NXT'],
    ETHE_BLOCK: [ETHE_BLOCK],
    BINA_CHART: []
}


JUN2016 = {
    GDAX_CHART: ['ETH-USD', 'ETH-BTC', 'BTC-USD', 'BTC-EUR'],
    POLO_CHART: ['BTC_BCN', 'BTC_BTS', 'BTC_BURST', 'BTC_DASH', 'BTC_DCR',
                 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2', 'BTC_ETH', 'BTC_FCT',
                 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NXT', 'BTC_SC',
                 'BTC_STR', 'BTC_VTC', 'BTC_XEM', 'BTC_XMR', 'BTC_XPM',
                 'BTC_XRP', 'USDT_BTC', 'USDT_DASH', 'USDT_ETH', 'USDT_LTC',
                 'USDT_NXT', 'USDT_STR', 'USDT_XMR', 'USDT_XRP', 'XMR_DASH',
                 'XMR_MAID', 'XMR_NXT'],
    POLO_TRADE: ['BTC_BCN', 'BTC_BTS', 'BTC_BURST', 'BTC_DASH', 'BTC_DCR',
                 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2', 'BTC_ETH', 'BTC_FCT',
                 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NXT', 'BTC_SC',
                 'BTC_STR', 'BTC_VTC', 'BTC_XEM', 'BTC_XMR', 'BTC_XPM',
                 'BTC_XRP', 'USDT_BTC', 'USDT_DASH', 'USDT_ETH', 'USDT_LTC',
                 'USDT_NXT', 'USDT_STR', 'USDT_XMR', 'USDT_XRP', 'XMR_DASH',
                 'XMR_MAID', 'XMR_NXT'],
    ETHE_BLOCK: [ETHE_BLOCK],
    BINA_CHART: []
}


TEST = {
    GDAX_CHART: ['ETH-BTC'],
    POLO_CHART: [],
    POLO_TRADE: [],
    BINA_CHART: [],
    ETHE_BLOCK: []
}

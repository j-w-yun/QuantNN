from datetime import datetime, timezone
import time
import dateutil.tz

import dateutil.parser as dp


def unix_to_utc(unix):
    return datetime.utcfromtimestamp(unix)


def utc_to_iso(utc):
    return utc.isoformat()


def unix_to_iso(unix):
    return utc_to_iso(unix_to_utc(unix))


def iso_to_unix(iso):
    return int(dp.parse(iso).replace(tzinfo=timezone.utc).timestamp())


def date_to_unix(date):
    return int(date.replace(tzinfo=timezone.utc).timestamp())


if __name__ == '__main__':
    t = time.time()
    print('Current UNIX : {}\n'.format(t))

    iso = unix_to_iso(1493000000)
    unix = iso_to_unix(iso)
    print('ISO  : {}'.format(iso))
    print('UNIX : {}\n'.format(unix))

    start_date_local = datetime(
        year=2017,
        month=8,
        day=1,
        hour=0,
        minute=0,
        tzinfo=dateutil.tz.tzlocal())
    # convert local tz datetime to unix
    utctz = dateutil.tz.tzutc()

    iso = utc_to_iso(start_date_local)
    print(iso)
    unix = iso_to_unix(iso)
    print(unix)

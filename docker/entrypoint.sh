#!/bin/sh
# Optionally throttle this container's network to emulate a slow WAN link,
# then drop root and run the command as the unprivileged `app` user.
#   NET_DELAY=20ms NET_RATE=100mbit   (needs cap_add: NET_ADMIN)
set -e

if [ -n "${NET_DELAY}${NET_RATE}" ]; then
    tc qdisc replace dev eth0 root netem ${NET_DELAY:+delay $NET_DELAY} ${NET_RATE:+rate $NET_RATE}
    echo "netem on eth0: delay=${NET_DELAY:-0ms} rate=${NET_RATE:-unlimited}"
fi

if [ "$(id -u)" = "0" ]; then
    exec setpriv --reuid=app --regid=app --init-groups "$@"
fi
exec "$@"

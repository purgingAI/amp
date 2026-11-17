#!/usr/bin/env bash
# HTTPS for the customer app on the LAN — so phones can use the microphone.
#
# Browsers only expose getUserMedia (the mic) in a SECURE context: https, or localhost.
# Opened as http://192.168.x.x the API does not exist at all, which is why voice "could not
# be used over the LAN". This makes a small local certificate authority once, and a server
# certificate for this laptop's current LAN addresses, localhost and <hostname>.local.
#
# On a phone, either tap through the browser's one-time warning, or install
# https://<ip>:8443/ca.crt as a trusted certificate for no warning at all.
# Re-run when the laptop's IP changes (run.sh does this automatically).
set -euo pipefail
cd "$(dirname "$0")/certs"
if [ ! -f ca.key ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.crt -days 825 \
    -subj "/CN=OpenWallet Demo Local CA" -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" 2>/dev/null
fi
IPS=$(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.' | grep -v '^172\.17\.' || true)
SAN="DNS:localhost,DNS:$(hostname).local,IP:127.0.0.1"
for ip in $IPS; do SAN="$SAN,IP:$ip"; done
openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.csr -subj "/CN=openwallet-demo" 2>/dev/null
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 397 \
  -extfile <(printf "subjectAltName=%s\nextendedKeyUsage=serverAuth\nkeyUsage=digitalSignature,keyEncipherment" "$SAN") 2>/dev/null
echo "$SAN" > san.txt
echo "certificate for: $SAN"

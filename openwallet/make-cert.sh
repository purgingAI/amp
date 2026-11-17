#!/usr/bin/env bash
# A certificate for the LAN address, so iOS will offer Face ID.
#
# Face ID on the web is WebAuthn, and WebAuthn is only exposed to a SECURE CONTEXT.
# http://192.168.x.x is not one, so Safari hides the API and the button cannot appear.
# This makes a self-signed certificate for whatever address the phone will type.
#
#   ./openwallet/make-cert.sh                 # for the current LAN IP
#   ./openwallet/make-cert.sh 127.0.0.1  # or name it explicitly
#
# Then:
#   OPENWALLET_TLS=1 CX_REQUIRE_PIN=1 ./openwallet/run.sh
#
# ON THE IPHONE, once — Safari will warn until all three are done:
#   1. open  http://<ip>:8440/dev-cert.pem   and let iOS download the profile
#   2. Settings → Profile Downloaded → Install
#   3. Settings → General → About → Certificate Trust Settings → turn it ON
#
# Step 3 is the one everybody misses. Without it the page is still "not private" and
# Face ID stays hidden, which looks like the feature is broken rather than untrusted.
set -euo pipefail
cd "$(dirname "$0")"

IP="${1:-}"
if [ -z "$IP" ]; then
  IP="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)"
fi
[ -n "$IP" ] || { echo "could not work out the LAN address — pass it: $0 127.0.0.1" >&2; exit 1; }

echo "  certificate for $IP (valid 825 days — Apple rejects longer)"
openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
  -keyout dev-key.pem -out dev-cert.pem \
  -subj "/CN=$IP/O=OpenWallet demo" \
  -addext "subjectAltName=IP:$IP,DNS:localhost,IP:127.0.0.1" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,digitalSignature,keyCertSign" \
  2>/dev/null

chmod 600 dev-key.pem
echo "  wrote openwallet/dev-cert.pem and dev-key.pem"
echo
echo "  start with:  OPENWALLET_TLS=1 CX_REQUIRE_PIN=1 ./openwallet/run.sh"
echo "  then open    https://$IP:8440/assist   on the phone"
echo
echo "  the phone needs the certificate trusted first — see the notes at the top of"
echo "  this script; the Certificate Trust Settings toggle is the step that is missed."

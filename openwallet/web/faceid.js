/* Face ID (and Touch ID) for the PIN sheet.
 *
 * ## What this does, and what it deliberately does not
 *
 * The wallet is the only thing that can say whether a PIN is correct. Face ID cannot tell
 * it anything — there is no "the face matched" the wallet API would accept. So this does
 * NOT replace the PIN check. It replaces the TYPING.
 *
 *   first payment   you enter the PIN · the wallet verifies it · we offer to remember it
 *   later payments  Face ID unlocks the remembered PIN · it goes to the wallet as normal
 *
 * Every payment is still authorised by the wallet, with the same wrong-PIN counting and
 * the same lockout. Nothing new is trusted: not this page, not the server, not a "verified"
 * flag that a client could set. The server never learns that Face ID was used, because it
 * does not need to and a claim it cannot check is worth nothing.
 *
 * ## Where the PIN lives
 *
 * On the device, encrypted with AES-GCM under a key the authenticator derives — the
 * WebAuthn **PRF extension**. That key does not exist until Face ID succeeds, and it is
 * never stored. Without a face, the stored bytes are unopenable.
 *
 * **If PRF is unavailable this refuses to enrol.** The fallback everyone reaches for —
 * keep the PIN in localStorage and merely *ask* for Face ID first — stores a payment
 * credential that anything with script access can read, while looking exactly as safe as
 * the real thing. A convenience feature is not worth a credential in the clear, so the
 * button simply does not appear.
 *
 * ## Why it may not appear at all
 *
 * WebAuthn needs a **secure context**: https, or localhost. Opened over plain http on a
 * LAN address — http://192.168.x.x:8440 — the browser does not expose it and there is no
 * workaround from this side. `faceIdWhyNot()` says which reason applies.
 */
(() => {
  const LS_CRED = 'openwallet.faceid.cred';      // credential id (not secret)
  const LS_BLOB = 'openwallet.faceid.blob';      // {iv, ct} — PIN sealed under the PRF key
  const RP_NAME = 'OpenWallet';
  const SALT = new TextEncoder().encode('openwallet-pin-v1');

  const b64u = {
    enc: b => btoa(String.fromCharCode(...new Uint8Array(b)))
      .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''),
    dec: s => Uint8Array.from(
      atob(s.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0)),
  };

  function supported() {
    return !!(window.PublicKeyCredential && navigator.credentials &&
              window.isSecureContext);
  }

  /** Why the button is not shown. Plain language — the reason is usually fixable. */
  window.faceIdWhyNot = function () {
    if (!window.isSecureContext)
      return 'Face ID needs a secure page (https). This demo is being served over plain ' +
             'http, so the browser will not offer it.';
    if (!window.PublicKeyCredential) return 'This browser has no Face ID support.';
    return '';
  };

  window.faceIdEnrolled = () => !!(localStorage.getItem(LS_CRED) &&
                                   localStorage.getItem(LS_BLOB));

  window.faceIdAvailable = async function () {
    if (!supported()) return false;
    try {
      return await PublicKeyCredential
        .isUserVerifyingPlatformAuthenticatorAvailable();
    } catch (_) { return false; }
  };

  async function keyFrom(prfBytes) {
    const base = await crypto.subtle.importKey('raw', prfBytes, 'HKDF', false,
                                               ['deriveKey']);
    return crypto.subtle.deriveKey(
      {name: 'HKDF', hash: 'SHA-256', salt: SALT, info: new TextEncoder().encode('pin')},
      base, {name: 'AES-GCM', length: 256}, false, ['encrypt', 'decrypt']);
  }

  /** Remember this PIN behind Face ID. Call ONLY after the wallet accepted it. */
  window.faceIdEnrol = async function (pin, userName) {
    if (!supported()) throw new Error(window.faceIdWhyNot());
    const challenge = crypto.getRandomValues(new Uint8Array(32));
    const userId = crypto.getRandomValues(new Uint8Array(16));
    const cred = await navigator.credentials.create({
      publicKey: {
        challenge, rp: {name: RP_NAME},
        user: {id: userId, name: userName || 'openwallet user',
               displayName: userName || 'OpenWallet'},
        pubKeyCredParams: [{type: 'public-key', alg: -7}, {type: 'public-key', alg: -257}],
        authenticatorSelection: {authenticatorAttachment: 'platform',
                                 userVerification: 'required',
                                 residentKey: 'preferred'},
        timeout: 60000,
        extensions: {prf: {}},
      },
    });
    const ext = cred.getClientExtensionResults();
    if (!ext || !ext.prf || !ext.prf.enabled) {
      // No PRF means no key that only a face can produce. Storing the PIN anyway would
      // put a payment credential where any script can read it.
      throw new Error('This device cannot protect the PIN with Face ID (no PRF support), ' +
                      'so I have not stored it. You can keep typing it.');
    }
    localStorage.setItem(LS_CRED, b64u.enc(cred.rawId));
    // Enrolment cannot read the PRF output, so seal the PIN with an immediate assertion.
    await seal(pin);
    return true;
  };

  async function assertPrf() {
    const id = localStorage.getItem(LS_CRED);
    if (!id) throw new Error('not enrolled');
    const assertion = await navigator.credentials.get({
      publicKey: {
        challenge: crypto.getRandomValues(new Uint8Array(32)),
        allowCredentials: [{type: 'public-key', id: b64u.dec(id)}],
        userVerification: 'required',
        timeout: 60000,
        extensions: {prf: {eval: {first: SALT}}},
      },
    });
    const ext = assertion.getClientExtensionResults();
    const first = ext && ext.prf && ext.prf.results && ext.prf.results.first;
    if (!first) throw new Error('Face ID did not return a key for this device.');
    return keyFrom(first);
  }

  async function seal(pin) {
    const key = await assertPrf();
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const ct = await crypto.subtle.encrypt({name: 'AES-GCM', iv}, key,
                                           new TextEncoder().encode(pin));
    localStorage.setItem(LS_BLOB, JSON.stringify({iv: b64u.enc(iv), ct: b64u.enc(ct)}));
  }

  /** Face ID, then the remembered PIN. Returns the PIN string. */
  window.faceIdUnlock = async function () {
    const raw = localStorage.getItem(LS_BLOB);
    if (!raw) throw new Error('not enrolled');
    const {iv, ct} = JSON.parse(raw);
    const key = await assertPrf();
    const out = await crypto.subtle.decrypt(
      {name: 'AES-GCM', iv: b64u.dec(iv)}, key, b64u.dec(ct));
    return new TextDecoder().decode(out);
  };

  /** Forget it. Called when the wallet rejects the stored PIN — see below. */
  window.faceIdForget = function () {
    localStorage.removeItem(LS_CRED);
    localStorage.removeItem(LS_BLOB);
  };
})();

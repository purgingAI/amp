# PayI — paid by AI (three-screen demo)

Two products on one protocol, VoicePay in both:

- **PayI** — pay people by voice through Jarvis ("send Alex 20 dirhams").
- **PalmPay** — walk into Falcon Café through the gate, then pay with your palm alone.

## Run

```bash
cd ~/amp && ./payi/run.sh
```

Run it on Sagar's laptop, which holds the TPM key and the webcam. It prints three URLs; open them in Chrome.

| Screen | URL | Who |
|---|---|---|
| 1 | `http://localhost:8420/payer` | Sagar. Use localhost so the mic and camera work. |
| 2 | `http://<ip>:8420/watch` | Watcher: PayI Core, every key, signature, ciphertext and ledger entry. |
| 3 | `http://<ip>:8420/wallet` | Alex. Click "Start receiving" once to enable voice and sound. |

Screens 2 and 3 open from other laptops on the same Wi-Fi.

Startup unseals the device key from the TPM, which takes 30–80 s. If the TPM refuses, the demo falls back to a software key and every screen says "Device key: software". For a quick rehearsal without the TPM, run `PAYI_SOFTWARE_KEY=1 ./payi/run.sh`.

## Demo script

1. **PayI, green.**
   1. Say or type "Send Alex 20 dirhams".
   2. Jarvis reads it back. **Touch the fingerprint sensor.** There's no "yes" and no confirm button; the touch approves exactly that payee and amount.
   3. The watcher shows the mandate, session keys, nonce, device signature, sealed channel, checks and ledger.
   4. Alex's wallet shows +20.
2. **PayI, amber.** Say "Send Alex 150 dirhams".
   1. It's above the voice-only cap.
   2. The same fingerprint touch is the second channel, bound to that exact payment.
3. **PayI, refusal.** Say "Send Vikram 20 dirhams". It's refused before anything is sent.
4. **PalmPay setup.** Switch to PalmPay and enrol your palm (webcam, or "Use demo hand").
5. **Walk past.** Nothing happens.
6. **Walk in.**
   1. The gate tracks you and your phone verifies the crossing.
   2. You get an anonymous pass.
   3. You can also drag "You" through the gate yourself.
7. **Pay at the till.**
   1. Go to the till and pick items.
   2. Click "Place palm on reader".
   3. The palm is released encrypted to till-1 only, matched 1:1 and deleted.
   4. The 16 acceptance checks run, then settlement.
   5. Falcon Café shows the payment on Alex's screen.
8. **Stranger.** Try a stranger's hand. It's refused.
9. **Queue.** Turn on "Shopper behind me". It's still your palm that pays, nearest phone first.
10. **Over the limit.** Select the catering order (290) twice. You're over the hands-free cap, so the phone refuses before your palm leaves it.
11. **Pay here.** Say "pay here" without a pass. That's explicit consent through VoicePay.
12. **Walk out.** The pass closes.

Reset gives you a fresh ledger (Sagar 1,000 AED).

## Honest limits

- UWB ranging is simulated from positions on the floor plan.
- The webcam measures hand geometry, not palm vein.
- The demo database is separate (`data/payi.db`).
- All cryptography, signatures, checks and ledger postings are real code paths:
  - `app/intent`, `app/monitor`, `app/mandate`, `app/ledger`
  - `proto/storepass`, `proto/biorelease`

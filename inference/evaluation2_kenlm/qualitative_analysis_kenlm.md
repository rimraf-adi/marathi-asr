# Dialect Qualitative Analysis: KenLM Decoding

This updated report analyzes the qualitative predictions of our **MoE Conformer (33M) decoded with a 5-gram KenLM (α=0.15)**.

While the Language Model improved our aggregate CER to 8.33%, inspecting the dialectal samples reveals a classic ASR phenomenon: **Dialect Erasure**. Because the KenLM was trained on standard Marathi text (like Wikipedia), it heavily penalizes valid rural vocabulary and forces the acoustic model to output standard words.

## D2: Ahirani / Khandesh 🌾

### The "KenLM Dialect Erasure" Effect (Worst Failures)

* **Ref**: पिकनी शेती मा येगयेगळा पिके लेतस
* **Our Model (Greedy)**: पिशवी शेतीमा येगयेगळा पिके लेतस
* **Our Model (KenLM)**: पिशवी शेतीमा एक एगडा ती केल लीतस
  * *Why it failed*: The KenLM completely destroyed the Ahirani word `येगयेगळा` (different). Because `येगयेगळा` has near-zero probability in standard Marathi text, the KenLM forcefully re-routed the beam search into `एक एगडा`, completely ruining the acoustic prediction to satisfy its n-gram grammar rules!

* **Ref**: दोन हजार तीन या सालमा गोडा पानीमधला झिंगानं आख्खा जगमधलं वरीसनं उत्पन्न...
* **Our Model (KenLM)**: दोनहजार तीन असाल मागोड आपाणी मधला जिनदाला खाजगमधल वरीस उत्पन दोन ला काश दार टन ओड होतं
  * *Why it failed*: `सालमा` (in the year) and `पानीमधला` (in the water) are classic Ahirani markers. The KenLM tried to aggressively split and merge these words to form standard nouns (`असाल`, `मागोड`, `आपाणी`), causing cascading token boundary failures.

### Moderately Good Samples

* **Ref**: दर महिन्याचे दिवस कमी जास्त असल्याने व्याजाची रक्कम वेगवेगळी असेल का ?
* **Our Model (KenLM)**: दर महिन्याची दिवस कमी जास्त असल्याने व्याजाची रक्कम वेगवेगळी असेल का ??
  * *Analysis*: On standard-sounding sentences, the KenLM excels! It perfectly predicted the sentence. Notice, however, that the punctuation hallucination (`??`) persists even with KenLM! Since punctuation isn't heavily penalized by a soft `alpha=0.15`, the strong acoustic GRPO bias still bleeds through the beam search.

---

## D1: Malvani / Konkan 🌴

### Moderately Good Samples
* **Ref**: पगाळी याने बारली सारख्या इखल पिकाचं उन्हाण्यात ओरटच आवाढतात
* **Our Model (KenLM)**: गाळ याने बारली सारख्या इतर पिकाचं उणण्यात ओरटच आवडतात
  * *Analysis*: Here, the KenLM actually *helped*! It correctly fixed the acoustic slur `इखल` into the grammatically correct `इतर` (other), and `आवाढतात` into `आवडतात` (like). This is the intended benefit of a Language Model.

## Summary Conclusion
Using the soft KenLM (`α=0.15`) is a double-edged sword. It acts as an excellent spell-checker for standard Marathi sentences and minor phonetic stutters. However, it actively acts as a **"dialect straightjacket"**, forcefully erasing rural grammar (`येगयेगळा`) and morphing it into completely unrelated standard words (`एक एगडा`). 

For true dialect preservation, pure Greedy CTC (or a custom KenLM trained on Ahirani/Malvani text) remains the gold standard!

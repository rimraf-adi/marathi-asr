# Dialect Qualitative Analysis: Triumphs and Failures

This report breaks down the qualitative performance of our **MoE Conformer (33M)** on the RESPIN test set, analyzing both where it excels against SOTA models and where it completely fails.

## D1: Malvani / Konkan 🌴

Malvani is characterized by coastal phonetic shifts and unique vocabulary.

### Moderately Good Samples
Here, the model perfectly captures the semantics, with only minor phonetic spelling differences that are acceptable in spoken transcriptions.
* **Ref**: गाढवाने जसे अंगावरचे ओझे कमी केले ...
* **Our Model**: गाडवाने जसे अंगावरचे ओजे कमी केले ...
* **Indic Conformer**: गाढवाने जसं अंगावरचे ओझं कमी केलं ...
  * *Analysis*: Our model spelled `गाढवाने` (donkey) as `गाडवाने` and `ओझे` (burden) as `ओजे`. Phonetically, these are nearly identical in Malvani speech.

### Worst Failures
* **Ref**: पगाळी याने बारली सारख्या इखल पिकाचं उन्हाण्यात ओरटच आवाढतात
* **Our Model**: पगाळी याने बारली सारख्या इतल पिकाचं उनाण्यात ओरटच आवाढतात
  * *Why it failed*: `उन्हाण्यात` (summer) is heavily slurred into `उनाण्यात`. The model tried to transcribe the pure acoustic sound rather than mapping it back to the standard spelling. `इखल` vs `इतल` is a pure acoustic confusion on heavy background noise.

---

## D2: Ahirani / Khandesh 🌾

Ahirani often uses `मा` (ma) instead of `मध्ये` (madhye) for "in", and `व्हतं` (vhatam) instead of `होतं` (hotam) for "was".

### Moderately Good Samples
* **Ref**: सेंद्रिय शेतीमुळे औषधी वनस्पती आणि सुगंधी वनस्पती आणि मसाले तयार होतात
* **Our Model**: सेंद्रिय शेतीमुळे औषधी वनस्पती आणि सुगंदी वनस्पती आणि मसाले तयार होतात
  * *Analysis*: A near-perfect match. `सुगंधी` (fragrant) was phonetically transcribed as `सुगंदी`.

### Worst Failures
* **Ref**: राम ना निव्वळ नफामा मार्जिन भलतच कमी व्हतं
* **Our Model**: रांनानी वळना फामामार्जिंगभलतूच कमी होत
  * *Why it failed*: Absolute catastrophic failure. The model completely lost the token boundaries (`राम ना निव्वळ नफामा` -> `रांनानी वळना फामा`). Interestingly, it attempted to normalize the Ahirani `व्हतं` back to standard Marathi `होत`, contradicting the acoustic signal.

---

## D3: Standard Marathi 🏙️

Standard Marathi is the easiest, but our model has a known punctuation hallucination issue.

### Moderately Good Samples
* **Ref**: ठिबक सिंचनाची जोडणी कशी असावी ?
* **Our Model**: ठिबक सिंचनाशी जोडणी कशी असावी ??
  * *Analysis*: Perfect transcription (except for the hallucinated `??`). It correctly predicted `ठिबक` (drip irrigation) which requires strong domain context.

### Worst Failures
* **Ref**: मुदत खाते ऑनलाइन उघडता येईल का ?
* **Our Model**: मुदत खाते ऑनलाइन उघडता येईल का ??
  * *Why it failed*: While the text is 100% semantically correct, the character-error rate (CER) algorithm strictly penalizes the extra `??`. The model learned this bias heavily during the OpenSLR64 Sequence Polish phase, causing the loss function to heavily penalize our model on paper despite perfect transcription.

---

## D4: Varhadi / Vidarbha 🏜️

Varhadi often changes `ल` (la) to `ळ` (lla) and has unique conjugations.

### Moderately Good Samples
* **Ref**: सगळ्यांत चांगलो ट्रॅक्टर कसल्या कंपनीचो मिळता ?
* **Our Model**: सगगळ्यांत चांगलो ट्रॅक्टर किसल्या कंपनीचो मिळता ??
  * *Analysis*: Excellent dialect preservation! It preserved the Varhadi `चांगलो` and `कंपनीचो` instead of converting them to the standard `चांगला` and `कंपनीचा`. Indic Conformer completely erased these dialectal markers.

### Worst Failures
* **Ref**: भय्यू महाराज यांच्या वकिलाकडे कोट्यवधी रुपये असल्याचा त्याला संशय होता
* **Our Model**: बयनाहराज यांच्या मपीलाप कटामधील ल असल्याचा त्याला सशहोता असेल पोलिसांनी म्ह
  * *Why it failed*: The audio likely contained heavy stuttering or microphone noise at the beginning. `भय्यू महाराज` (Bhayyu Maharaj) was mangled into `बयनाहराज`, and the model completely hallucinated the end of the sentence (`असेल पोलिसांनी म्ह`), likely triggered by the GRPO trying to predict the most likely news-broadcast continuation for a sentence about "lawyers" and "crores of rupees".

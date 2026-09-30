# Qualitative Error Analysis Report: `KATHBATH_NOISY` (Marathi)

### Distribution Summary (1631 total utterances)
- **Easy Sentences (WER ≤ 15%)**: 45 utterances (2.8%)
- **Medium Difficulty (15% < WER ≤ 50%)**: 418 utterances (25.6%)
- **Hard / Failed Sentences (WER > 50%)**: 1168 utterances (71.6%)

---

## 1. Easy Sentences (Near-Perfect / Exact Matches)

These utterances feature standard grammatical structure, clear speech articulation, common Marathi vocabulary, and minimal numeric or loanword interference.

### Example E-1 (`sample_00094`)
- **Reference**: `त्यामुळे लग्नातील खर्चामध्ये कपात करणे किंवा वैध मार्गांनं कर्ज घेणे हा पर्याय त्यांच्याकडे उपलब्ध होता`
- **Greedy CTC**: `सत्यामुळे लगनातील खर्चामध्ये कपात करणे किंवा वैद्य मार्गाना कर्ज घेणे हा पर्यात्यांच्याकडे उपलब्ध होता` *(WER: 40.0%, CER: 8.7%)*
- **Lexicon Beam**: `सत्यामुळे लगनातील खर्चामध्ये कपात करणे किंवा वैद्य मार्गाना कर्ज घेणे हा पर्यात्यांच्याकडे उपलब्ध होता` *(WER: 40.0%, CER: 8.7%)*
- **KenLM Default (α=0.5)**: `त्यामुळे लग्नातील खर्चामध्ये कपात करणे किंवा वैद्यमार्गाना कर्ज घेणे हा पर्यात्यांच्याकडे उपलब्ध होता` *(WER: 26.7%, CER: 6.8%)*
- **KenLM Soft (α=0.15)**: `त्यामुळे लग्नातील खर्चामध्ये कपात करणे किंवा वैद्य मार्गान कर्ज घेणे हा पर्याय त्यांच्याकडे उपलब्ध होता` *(WER: 13.3%, CER: 4.8%)*
- **SraVaani 1.0**: `त्यामुळे लग्नातील खर्चामध्ये कपात करणे किंवा वैद्य मार्गानं कर्ज घेणे हा पर्याय त्यांच्याकडे उपलब्ध होता` *(WER: 13.3%)*
- **Indic Conformer**: `त्यामुळे लग्नातील खर्चामध्ये कपात करणे किंवा वैद्यमार्गानें कर्ज घेणे हा पर्याय त्यांच्याकडे उपलब्ध होता` *(WER: 13.3%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

### Example E-2 (`sample_00170`)
- **Reference**: `त्यामुळे सरकारी कामकाजात अडथळा आणल्या प्रकरणी त्यांच्यावर गुन्हा दाखल करण्यात आला आहे`
- **Greedy CTC**: `त्यामुळे सरकारी कामका चत अडथळा आणिल्या प्रकाणी त्ांच्यावल गुन्हा गाखल करण्यात ला आहे` *(WER: 58.3%, CER: 9.4%)*
- **Lexicon Beam**: `त्यामुळे सरकारी कामका जत अडथळा आणिल्या प्रकाणी त्ांच्यावल गुन्हा गाखल करण्यात ला आहे` *(WER: 58.3%, CER: 9.4%)*
- **KenLM Default (α=0.5)**: `त्यामुळे सरकारी कामकाज अडथळा आणल्या प्रकरण त्याच्यावर गुन्हा दाखल करण्यात आला आहे` *(WER: 25.0%, CER: 4.7%)*
- **KenLM Soft (α=0.15)**: `त्यामुळे सरकारी कामकाजात अडथळा आणल्या प्रकरणी त्यांच्यावर गुन्हा दाखल करण्यात आला आहे` *(WER: 0.0%, CER: 0.0%)*
- **SraVaani 1.0**: `त्यामुळे सरकारी कामकाजात अडथळा आणल्याप्रकरणी त्यांच्यावर गुन्हा दाखल करण्यात आला आहे` *(WER: 16.7%)*
- **Indic Conformer**: `त्यामुळे सरकारी कामकाजात अडथळा आणल्याप्रकरणी त्यांच्यावर गुन्हा दाखल करण्यात आला आहे` *(WER: 16.7%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

### Example E-3 (`sample_00205`)
- **Reference**: `बाजाराच्या दिवशी शेतकरी व व्यापारी भाजीपाला विक्रीसाठी आणतात`
- **Greedy CTC**: `बाजाराच्या दिवशी शेतकरी व व्यापारी भाजीपाला विक्रीसाठी आणतात` *(WER: 0.0%, CER: 0.0%)*
- **Lexicon Beam**: `बाजाराच्या दिवशी शेतकरी व व्यापारी भाजीपाला विक्रीसाठी आणतात` *(WER: 0.0%, CER: 0.0%)*
- **KenLM Default (α=0.5)**: `बाजाराच्या दिवशी शेतकरी वा व्यापारी भाजीपाला विक्रीसाठी आणतात` *(WER: 12.5%, CER: 1.7%)*
- **KenLM Soft (α=0.15)**: `बाजाराच्या दिवशी शेतकरी व व्यापारी भाजीपाला विक्रीसाठी आणतात` *(WER: 0.0%, CER: 0.0%)*
- **SraVaani 1.0**: `बाजाराच्या दिवशी शेतकरी व व्यापारी भाजीपाला विक्रीसाठी आणतात` *(WER: 0.0%)*
- **Indic Conformer**: `बाजाराच्या दिवशी शेतकरी व व्यापारी भाजीपाला विक्रीसाठी आणतात` *(WER: 0.0%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

### Example E-4 (`sample_00212`)
- **Reference**: `तो एक रंग एक नवीन वर्षाचे सौंदर्य व्यवस्था करणे शक्य आहे`
- **Greedy CTC**: `तो एक रंग एक नवीन वर्षाचे संदर्य व्यवस्था करणे शक्य आहे` *(WER: 9.1%, CER: 1.8%)*
- **Lexicon Beam**: `तो एक रंग एक नवीन वर्षाचे संदर्य व्यवस्था करणे शक्य आहे` *(WER: 9.1%, CER: 1.8%)*
- **KenLM Default (α=0.5)**: `तो एक रंग एक नवीन वर्षाचे सौंदर्य व्यवस्था करणे शक्य आहे` *(WER: 0.0%, CER: 0.0%)*
- **KenLM Soft (α=0.15)**: `तो एक रंग एक नवीन वर्षाचे सौंदर्य व्यवस्था करणे शक्य आहे` *(WER: 0.0%, CER: 0.0%)*
- **SraVaani 1.0**: `तो एक रंग एक नवीन वर्षाचे सौंदर्य व्यवस्था करणे शक्य आहे` *(WER: 0.0%)*
- **Indic Conformer**: `तो एक रंग एक नवीन वर्षाचे सौंदर्य व्यवस्था करणे शक्य आहे` *(WER: 0.0%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

### Example E-5 (`sample_00229`)
- **Reference**: `म्हणूनच मी त्यांचं नाव वगैरे आताच काही सांगत नाहीय`
- **Greedy CTC**: `महनूनस्मी त्यांच नाव वगेरे आताच काही सांगत नाही` *(WER: 55.6%, CER: 14.0%)*
- **Lexicon Beam**: `महनूनस्मी त्यांच नाव वगेरे आताच काही सांगत नाही` *(WER: 55.6%, CER: 14.0%)*
- **KenLM Default (α=0.5)**: `म्हणूनच मी त्यांचं नाव वगैरे आतच काहीही सांगत नाही` *(WER: 33.3%, CER: 8.0%)*
- **KenLM Soft (α=0.15)**: `म्हणूनच मी त्यांचं नाव वगैरे आताच काही सांगत नाही` *(WER: 11.1%, CER: 2.0%)*
- **SraVaani 1.0**: `म्हणूनच मी त्यांचं नाव वगैरे आत्ताच काही सांगत नाही` *(WER: 22.2%)*
- **Indic Conformer**: `म्हणूनच मी त्यांचं नाव वगैरे आत्ताच काही सांगत नाही` *(WER: 22.2%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

---

## 2. Medium Difficulty Sentences (Partial Errors: 15% < WER ≤ 50%)

In this bucket, acoustic recognition is mostly correct, but errors occur due to postposition attachment, minor spelling variations, or spoken vs. written numeral differences.

### Example M-1 (`sample_00003`)
- **Reference**: `राने तयार असल्याने आता केवळ टोकण करण्याचे काम शिल्लक राहिले आहे`
- **Greedy CTC**: `राणे तयार असल्याने आतककेवळ टोकन करण्याचे कम शिल्लक राहीलली आहे` *(WER: 54.5%)*
- **Lexicon Beam**: `राणे तयार असल्याने आतक्केवळ टोकन करण्याचे कम शिल्लक राही लली आहे` *(WER: 63.6%)*
- **KenLM Default (α=0.5)**: `राणी तयार असल्याने आता केवळ टोकन करण्याचे का शिल्लक राहिले आहेत` *(WER: 36.4%)*
- **KenLM Soft (α=0.15)**: `राणे तयार असल्याने आता केवळ टोकन करण्याचे का शिल्लक राहिले आहे` *(WER: 27.3%)*
- **SraVaani 1.0**: `राणे तयार असल्याने आता केवळ टोकन करण्याचे काम शिल्लक राहिलेले आहे` *(WER: 27.3%)*
- **Indic Conformer**: `राणे तयार असल्याने आता केवळ टोकण करण्याचे काम शिल्लक राहिलेले आहे` *(WER: 18.2%)*
- **Failure Causes**: Phonetic Substitution / Unseen Vocabulary
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

### Example M-2 (`sample_00004`)
- **Reference**: `दीर्घ पल्ल्याचे आलाप घेण्याची क्षमता यामुळे त्यांच्या गायनाला रसिकांची दाद मिळाली`
- **Greedy CTC**: `दीर्घ पल्यांची आलाभ घेण्याचीक्षमता यामुळे त्यांच्या गायनाला रसिकांचे दाग मिळाली` *(WER: 54.5%)*
- **Lexicon Beam**: `दीर्घ पल्यांची आलाभ घेण्याचीक्षमता यामुळे त्यांच्या गायनाला रसिकांचे दाग मिळाली` *(WER: 54.5%)*
- **KenLM Default (α=0.5)**: `दीर्घ पल्ांची,आलाभ घेण्याची क्षमता यामुळे त्यांच्या गायनालारसिकांचे दाग मिळाली` *(WER: 45.5%)*
- **KenLM Soft (α=0.15)**: `दीर्घ पल्यांची आला घेण्याची क्षमता यामुळे त्यांच्या गायनाला रसिकांचे दाग मिळाली` *(WER: 36.4%)*
- **SraVaani 1.0**: `दीर्घपल्ल्यांची आलाप घेण्याची क्षमता यामुळे त्यांच्या गायनाला रसिकांची दाद मिळाली` *(WER: 18.2%)*
- **Indic Conformer**: `दीर्घ पल्ल्यांची आलाप घेण्याची क्षमता यामुळे त्यांच्या गायनाला रसिकांची दाद मिळाली` *(WER: 9.1%)*
- **Failure Causes**: Phonetic Substitution / Unseen Vocabulary
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

### Example M-3 (`sample_00007`)
- **Reference**: `अशा सर्व परिस्थितीत केंद्राने आता महानगरपालिकेला या योजनेतून अनुदान देणे कमी केले आहे`
- **Greedy CTC**: `शा सर्वा परिस्थितीत केंद्राने आता महानगरपाली केला या योजनेतून अनुदान देणे कमी केले आहेत` *(WER: 38.5%)*
- **Lexicon Beam**: `शा सर्वा परिस्थितीत केंद्राने आता महानगरपाली केला या योजनेतून अनुदान देणे कमी केले आहेत` *(WER: 38.5%)*
- **KenLM Default (α=0.5)**: `अशा सर्व परिस्थितीत केंद्राने आता महानगरपालिकेला या योजनेतून अनुदान देणे कमी केले आहेत` *(WER: 7.7%)*
- **KenLM Soft (α=0.15)**: `अशा सर्व परिस्थितीत केंद्राने आता महानगरपालि केला या योजनेतून अनुदान देणे कमी केले आहेत` *(WER: 23.1%)*
- **SraVaani 1.0**: `अशा सर्व परिस्थितीत केंद्राने आता महानगरपालिकेला या योजनेतून अनुदान देणे कमी केले आहे` *(WER: 0.0%)*
- **Indic Conformer**: `अशा सर्व परिस्थितीत केंद्राने आता महानगरपालिकेला या योजनेतून अनुदान देणे कमी केले आहेत` *(WER: 7.7%)*
- **Failure Causes**: Phonetic Substitution / Unseen Vocabulary
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

### Example M-4 (`sample_00010`)
- **Reference**: `एखाद्या कलेचं परीक्षण करण्यासाठी त्या कलेची जाण असण्यापेक्षा त्याचा अभ्यास असणं महत्त्वाचं आहे`
- **Greedy CTC**: `एखादा कलेच परीक्षण करण्यासाठी त्याकलेच जान असण्यापेक्षा त्याचा अपल्यास असण महत्त्वाचो आहे` *(WER: 61.5%)*
- **Lexicon Beam**: `एखादा कलेच परीक्षण करण्यासाठी त्याकलेच जान असण्यापेक्षा त्याचा अपल्यास असण महत्त्वाचो आहे` *(WER: 61.5%)*
- **KenLM Default (α=0.5)**: `एखादा कलेचपरीक्षण करण्यासाठी त्या कलेचजान असण्यापेक्षा त्याचा अभ्यास असणं महत्त्वाचा आहे` *(WER: 46.1%)*
- **KenLM Soft (α=0.15)**: `एखादा कलेच परीक्षण करण्यासाठी त्या कलेच जान असण्यापेक्षा त्याचा अभ्यास असणं महत्त्वाचा आहे` *(WER: 38.5%)*
- **SraVaani 1.0**: `एखाद्या कलेचं परीक्षण करण्यासाठी त्या कलेचं जाण असण्यापेक्षा त्याचा अभ्यास असणं महत्त्वाचं आहे` *(WER: 7.7%)*
- **Indic Conformer**: `एखाद्या कलेचं परीक्षण करण्यासाठी त्या कलेचं जाण असण्यापेक्षा त्याचा अभ्यास असणं महत्त्वाचं आहे` *(WER: 7.7%)*
- **Failure Causes**: Anusvara / Nasalization Variant
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

### Example M-5 (`sample_00011`)
- **Reference**: `गर्भारपणात यकृताचे आजार का व कसे होऊ शकतात ते जाणून घेऊया`
- **Greedy CTC**: `गरभारपरणात यकृताचे आजारका व कसे होऊ शकता ते जाणून घेवरया` *(WER: 45.5%)*
- **Lexicon Beam**: `गरभारपरणात यकृताचे आजार का व कसे होऊ शकता ते जाणून घेवरया ?` *(WER: 27.3%)*
- **KenLM Default (α=0.5)**: `गरभारपरणात याचे आजार का वर कसे होऊ शकतात ते जाणून गेला` *(WER: 36.4%)*
- **KenLM Soft (α=0.15)**: `गरभार पानात याचे आजार का वर कसे होऊ शकतात ते जाणून घेव` *(WER: 45.5%)*
- **SraVaani 1.0**: `गर्भारपणात यकृताचे आजार का व कसे होऊ शकतात ते जाणून घेऊया` *(WER: 0.0%)*
- **Indic Conformer**: `गर्भारपणात यकृताचे आजार का व कसे होऊ शकतात ते जाणून घेऊया` *(WER: 0.0%)*
- **Failure Causes**: Phonetic Substitution / Unseen Vocabulary
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

---

## 3. Hard & Failed Sentences (Severe Discrepancies: WER > 50%)

These sentences highlight the principal failure boundaries of zero-shot acoustic transfer and greedy CTC.

### Example H-1 (`sample_00000`)
- **Reference**: `१३ वर्षाची असताना तिला मिनी राष्ट्रीय बॅडमिंटन चॅम्पियनशिप जिंकली होती`
- **Greedy CTC**: `तेरा वर्षांची अस्तनातीला मीनी राष्ट्रीय बाटमेंटनच्या पेलशि जिटली होती` *(WER: 80.0%)*
- **Lexicon Beam**: `तेरा वर्षाची अस्तनातीला मीनी राष्ट्रीय बाटमेंटनच्या पेलशि जिटली होती` *(WER: 70.0%)*
- **KenLM Default (α=0.5)**: `तेरा वर्षांची असताना तिला मिनी राष्ट्रीय बाटमेंटनच्या पेशी जिकली होती` *(WER: 50.0%)*
- **KenLM Soft (α=0.15)**: `तेरा वर्षांची असताना तिला मिनी राष्ट्रीय बाँड मेंटन च्या पेशी जंगली होती` *(WER: 70.0%)*
- **SraVaani 1.0**: `तेरा वर्षाची असताना तिला मिनी राष्ट्रीय बॅडमिंटन चॅम्पियनशिप जिंकली होती` *(WER: 10.0%)*
- **Indic Conformer**: `तेरा वर्षांची असताना तिला मिनी राष्ट्रीय बॅडमिंटन चॅम्पियनशिप जिंकली होती` *(WER: 20.0%)*
- **Root Cause Analysis**: **Numeral Mismatch (Digits vs Spoken Words), English Loanword / Novel Phonotactics**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.

### Example H-2 (`sample_00001`)
- **Reference**: `त्यामुळे रिक्षा चालकाने थेट नवी मुंबई पोलीस आयुक्तांकडे यासंदर्भात तक्रार केली`
- **Greedy CTC**: `तामुळे रिक्षा चालताने ्थेट नवी मोंबईल पुलीचा एक् तमकडे या सनदर्भात तकरारली` *(WER: 90.9%)*
- **Lexicon Beam**: `तामुळे रिक्षा चालताने थेट नवी मोंबईल पुलीचा एक्तमकडे या सनदर्भात तकरारली` *(WER: 72.7%)*
- **KenLM Default (α=0.5)**: `त्यामुळे रिक्षा चालकाने थेट नवी मुंबई पोलीस एकदम कडे या संदर्भात तक्रारी` *(WER: 45.5%)*
- **KenLM Soft (α=0.15)**: `त्यामुळे रिक्षा चालकाने थेट नवी मुंबई पुलीचा एक तन कडे या संदर्भात तक्रार ली` *(WER: 63.6%)*
- **SraVaani 1.0**: `त्यामुळे रिक्षाचालकाने थेट नवी मुंबई पोलीस आयुक्तांकडे यासंदर्भात तक्रार केली` *(WER: 18.2%)*
- **Indic Conformer**: `त्यामुळे रिक्षाचालकाने थेट नवी मुंबई पोलीस आयुक्तांकडे यासंदर्भात तक्रार केली` *(WER: 18.2%)*
- **Root Cause Analysis**: **Anusvara / Nasalization Variant**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.

### Example H-3 (`sample_00002`)
- **Reference**: `हा चित्रपट वास्तववादी आणि त्याचबरोबर व्यावसायिकही आहे असं बोललं जातंय`
- **Greedy CTC**: `हा चिक्रफट बास्तभादी आणि त्यास बरबर व्यवसायिकही आहे अस बोल जाते` *(WER: 80.0%)*
- **Lexicon Beam**: `हा चिक्राफट बास्तभादी आणि त्यास बरबर व्यवसायिकही आहे अस बोल जाते` *(WER: 80.0%)*
- **KenLM Default (α=0.5)**: `हा चित्रपट वास्तववादी आणि त्याचबरोबर व्यवसायिक ही आहे अस बोललो जाते` *(WER: 50.0%)*
- **KenLM Soft (α=0.15)**: `हा चित्रपट वास्तववादी आणि त्याच बरोबर व्यवसायिक ही आहे अस बोललो जाते` *(WER: 70.0%)*
- **SraVaani 1.0**: `हा चित्रपट वास्तववादी आणि त्याचबरोबर व्यावसायिकही आहे असं बोललं जातंय` *(WER: 0.0%)*
- **Indic Conformer**: `हा चित्रपट वास्तववादी आणि त्याचबरोबर व्यावसायिकही आहे असं बोललं जातं` *(WER: 10.0%)*
- **Root Cause Analysis**: **Anusvara / Nasalization Variant**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.

### Example H-4 (`sample_00005`)
- **Reference**: `वेलिंग्टन कसोटीत भारताच्या खेळाडूंना वाऱ्यामुळे आणि उसळी घेणाऱ्या चेंडूंचा सामना करावा लागला होता`
- **Greedy CTC**: `्रमिंटनकसोटी भारताच्या केाडून वार्यामुळे कसरी घेणा्या चेंदूका ामदा कराबा लागला होता` *(WER: 76.9%)*
- **Lexicon Beam**: `्रमिंटनकसोटी भारताच्या केाडून वार्यामुळे कसरी घेणा⁇्या चेंदूका मदा कराबा लागला होता` *(WER: 76.9%)*
- **KenLM Default (α=0.5)**: `रिटनसोटी,भारताच्या कडून वार्यामुळे कसली घेणारा चेंडू मना करावा लागला होता` *(WER: 69.2%)*
- **KenLM Soft (α=0.15)**: `प्रिंट कसोटी भारताच्या फेडून वार्यामुळे कसली घेणारा चेंडू का आमना करावा लागला होता` *(WER: 69.2%)*
- **SraVaani 1.0**: `वॅडिंग्टन कचोटीत भारतातच्या खेळाडूंना वाऱ्यामुळे आणि उसळी घेणाऱ्या चेंडूंचा सामना करावा लागला होता` *(WER: 38.5%)*
- **Indic Conformer**: `वेलिंग्टन कसोटीत भारताच्या खेळाडूंना वाऱ्यामुळे आणिध उसळी घेणाऱ्या चेंडूंचा सामना करावा लागला होता` *(WER: 23.1%)*
- **Root Cause Analysis**: **Anusvara / Nasalization Variant**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.

### Example H-5 (`sample_00006`)
- **Reference**: `दर अर्ध्या तासाने मुलांना डोळ्याचे व्यायाम करायला सांगायला हवेत`
- **Greedy CTC**: `दर अर्ध्या काकांनी मुलांना डोळयाने व्याळाम करायला सांगायला हवे` *(WER: 44.4%)*
- **Lexicon Beam**: `दर अर्ध्या काकांनी मुलांना डोळयाने व्याळाम करायला सांगायला हवे` *(WER: 44.4%)*
- **KenLM Default (α=0.5)**: `दर अर्ध्या काकांनी मुलांना डोळा ते व्यायाम करायला सांगाल हवे` *(WER: 55.6%)*
- **KenLM Soft (α=0.15)**: `दर अर्ध्या काकांनी मुलांना डोळा ते व्यायाम करायला सांगाल हवे` *(WER: 55.6%)*
- **SraVaani 1.0**: `दर अर्ध्या तासांनी मुलांना डोळ्याचे व्यायाम करायला सांगायला हवेत` *(WER: 11.1%)*
- **Indic Conformer**: `दर अर्ध्या तासांनी मुलांना डोळ्याचे व्यायाम करायला सांगायला हवेत` *(WER: 11.1%)*
- **Root Cause Analysis**: **Phonetic Substitution / Unseen Vocabulary**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.


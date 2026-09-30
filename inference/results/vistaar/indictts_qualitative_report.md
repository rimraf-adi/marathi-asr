# Qualitative Error Analysis Report: `INDICTTS` (Marathi)

### Distribution Summary (100 total utterances)
- **Easy Sentences (WER ≤ 15%)**: 12 utterances (12.0%)
- **Medium Difficulty (15% < WER ≤ 50%)**: 44 utterances (44.0%)
- **Hard / Failed Sentences (WER > 50%)**: 44 utterances (44.0%)

---

## 1. Easy Sentences (Near-Perfect / Exact Matches)

These utterances feature standard grammatical structure, clear speech articulation, common Marathi vocabulary, and minimal numeric or loanword interference.

### Example E-1 (`sample_00014`)
- **Reference**: `कोणी मला खाल्ले तर त्याचे पोट देखील भरणार नाही.`
- **Greedy CTC**: `कोळी मला खाल्ले तर त्याचे पोठ देखील भरणार नाही` *(WER: 22.2%, CER: 4.3%)*
- **Lexicon Beam**: `कोळी मला खाल्ले तर त्याचे पोठ देखील भरणार नाही` *(WER: 22.2%, CER: 4.3%)*
- **KenLM Default (α=0.5)**: `कोणी मला खाल्ले तर त्याचे पोट देखील भरणारनाही` *(WER: 22.2%, CER: 2.2%)*
- **KenLM Soft (α=0.15)**: `कोणी मला खाल्ले तर त्याचे पोट देखील भरणार नाही` *(WER: 0.0%, CER: 0.0%)*
- **SraVaani 1.0**: `कोणी मला खाल्ले तर त्याचे पोट देखील भरणार नाही` *(WER: 0.0%)*
- **Indic Conformer**: `कोणी मला खाल्ले तर त्याचे पोटदेखील भरणार नाही` *(WER: 22.2%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

### Example E-2 (`sample_00017`)
- **Reference**: `मला काहीच माहीत नसताना हा माझा मुलगा म्हणतेस`
- **Greedy CTC**: `मला काहीच माहित नसताना हा माझा मुलगा म्हणतेस` *(WER: 12.5%, CER: 2.3%)*
- **Lexicon Beam**: `मला काहीच माहित नसताना हा माझा मुलगा म्हणतेस` *(WER: 12.5%, CER: 2.3%)*
- **KenLM Default (α=0.5)**: `मला काहीच माहित नसताना हा माझा मुलगा म्हणत` *(WER: 25.0%, CER: 6.8%)*
- **KenLM Soft (α=0.15)**: `मला काहीच माहित नसताना हा माझा मुलगा म्हणतेस` *(WER: 12.5%, CER: 2.3%)*
- **SraVaani 1.0**: `मला काहीच माहीत नसताना हा माझा मुलगा म्हणतेच` *(WER: 12.5%)*
- **Indic Conformer**: `मला काहीच माहीत नसताना हा माझा मुलगा म्हणतेस` *(WER: 0.0%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

### Example E-3 (`sample_00021`)
- **Reference**: `वसुबारस ह्याचा अर्थ- वसु म्हणजे द्रव्य, धन, त्यासाठी असलेली बारस म्हणजे द्वादशी.`
- **Greedy CTC**: `बशुबारस ह्याचा अर्ठ बसू म्हणजे द्रब्य धन त्यासाठी असलेली बारस म्हणजे द्वादशे` *(WER: 41.7%, CER: 9.2%)*
- **Lexicon Beam**: `बशुबारस ह्याचा अर्ठ बसू म्हणजे द्रब्य धन त्यासाठी असलेली बारस म्हणजे द्वादशे` *(WER: 41.7%, CER: 9.2%)*
- **KenLM Default (α=0.5)**: `वसुबारस याचा अर्थ वसु म्हणजे द्रव्य धन त्यासाठी असलेली बारस म्हणजे द्वादशी` *(WER: 8.3%, CER: 2.6%)*
- **KenLM Soft (α=0.15)**: `वसुबारस ह्याचा अर्थ वसु म्हणजे द्रव्य धन त्यासाठी असलेली बारस म्हणजे द्वादशे` *(WER: 8.3%, CER: 1.3%)*
- **SraVaani 1.0**: `वसु बारस ह्याचा अर्थ वसू म्हणजे द्रव्य धन त्यासाठी असलेली बारस म्हणजे द्वादशी` *(WER: 25.0%)*
- **Indic Conformer**: `वसुबारस ह्याचा अर्थ वसू म्हणजे द्रव्य धन त्यासाठी असलेली बारस म्हणजे द्वादशी` *(WER: 8.3%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

### Example E-4 (`sample_00032`)
- **Reference**: `मी अगोदर पुढे गेलो तर त्याचा किती प्रकारांनी लाभ होईल!`
- **Greedy CTC**: `मी अगूदर पोठे गेलो तर त्याचा किती प्रकारांनी लाभ होई ?` *(WER: 30.0%, CER: 7.5%)*
- **Lexicon Beam**: `मी अगूदर पोठे गेलो तर त्याचा किती प्रकारांनी लाभ होई ?` *(WER: 30.0%, CER: 7.5%)*
- **KenLM Default (α=0.5)**: `मी अगोदर पुढे गेलो तर त्याचा किती प्रकारांनी लाभ होई ?` *(WER: 10.0%, CER: 1.9%)*
- **KenLM Soft (α=0.15)**: `मी अगोदर पुढे गेलो तर त्याचा किती प्रकारांनी लाभ होई ?` *(WER: 10.0%, CER: 1.9%)*
- **SraVaani 1.0**: `मी अगोदर पुढे गेलो तर त्याचा किती प्रकारांनी लाभ होईल` *(WER: 0.0%)*
- **Indic Conformer**: `मी अगोदर पुढे गेलो तर त्याचा किती प्रकारांनी लाभ होईल` *(WER: 0.0%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

### Example E-5 (`sample_00048`)
- **Reference**: `संध्याकाळ होत आल्यामुळे खाली उतरणे आवश्यक होते.`
- **Greedy CTC**: `सं्ाकाळ होत आल्यामुळे खाली उतरणे आवश्यक होते` *(WER: 14.3%, CER: 4.3%)*
- **Lexicon Beam**: `सं्ाकाळ होत आल्यामुळे खाली उतरणे आवश्यक होते` *(WER: 14.3%, CER: 4.3%)*
- **KenLM Default (α=0.5)**: `संध्याकाळहोत आल्यामुळे खाली उतरणे आवश्यक होते` *(WER: 28.6%, CER: 2.2%)*
- **KenLM Soft (α=0.15)**: `संध्याकाळ होत आल्यामुळे खाली उतरणे आवश्यक होते` *(WER: 0.0%, CER: 0.0%)*
- **SraVaani 1.0**: `संध्याकाळ होत आल्यामुळे खाली उतरणे आवश्यक होते` *(WER: 0.0%)*
- **Indic Conformer**: `संध्याकाळ होत आल्यामुळे खाली उतरणे आवश्यक होते` *(WER: 0.0%)*
- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.

---

## 2. Medium Difficulty Sentences (Partial Errors: 15% < WER ≤ 50%)

In this bucket, acoustic recognition is mostly correct, but errors occur due to postposition attachment, minor spelling variations, or spoken vs. written numeral differences.

### Example M-1 (`sample_00003`)
- **Reference**: `ह्यात काही विशेष असेल तर सांगण्याची कृपा करावी.`
- **Greedy CTC**: `यात काही विशेष असेल तर चांगण्याची रूपा करावी` *(WER: 37.5%)*
- **Lexicon Beam**: `यात काही विशेष असेल तर चांगण्याची रूपा करावी` *(WER: 37.5%)*
- **KenLM Default (α=0.5)**: `यात काही विशेष असेल तर सांगण्याची कृपा करावे` *(WER: 25.0%)*
- **KenLM Soft (α=0.15)**: `यात काही विशेष असेल तर सांगण्याची रुपया करावी` *(WER: 25.0%)*
- **SraVaani 1.0**: `यात काही विशेष असेल तर सांगण्याची कृपा करावी` *(WER: 12.5%)*
- **Indic Conformer**: `यात काही विशेष असेल तर सांगण्याची कृपा करावी` *(WER: 12.5%)*
- **Failure Causes**: Phonetic Substitution / Unseen Vocabulary
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

### Example M-2 (`sample_00005`)
- **Reference**: `ह्या झाडाची क्विना-क्विना नावाची साले विषारी मानली जातात.`
- **Greedy CTC**: `ह्या झाडाची क्विना कविणा नावाची साले विषारी मानली जातात` *(WER: 11.1%)*
- **Lexicon Beam**: `ह्या झाडाची क्विना विणा नावाची साले विषारी मानली जातात` *(WER: 11.1%)*
- **KenLM Default (α=0.5)**: `ह्या झाडाची विना विना नावाची साल विषारी मानली जातात` *(WER: 33.3%)*
- **KenLM Soft (α=0.15)**: `ह्या झाडाची विना विना नावाची चाले विषारी मानली जातात` *(WER: 33.3%)*
- **SraVaani 1.0**: `ह्या झाडाची क्विना क्विना नावाची साले विषारी मानली जातात` *(WER: 0.0%)*
- **Indic Conformer**: `ह्या झाडाची क्विना क्विना नावाची साले विषारी मानली जातात` *(WER: 0.0%)*
- **Failure Causes**: Phonetic Substitution / Unseen Vocabulary
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

### Example M-3 (`sample_00008`)
- **Reference**: `त्याला नदी पार करायची होती तेवढयात त्याला घोडयांच्या टापांचा आवाज ऐकू आला`
- **Greedy CTC**: `त्याला नधी पारकारायचे होती पेवढ्यात त्याला घोड्यांच्या टापांचा आभा सयकवाला` *(WER: 66.7%)*
- **Lexicon Beam**: `त्याला नधी पारकारायचे होती पेवढ्यात त्याला घोड्यांच्या टापांचा आभा ायकवाला` *(WER: 66.7%)*
- **KenLM Default (α=0.5)**: `त्याला नदी पार करायची होती तेवढा त्याला घोड्यांच्या टापांचा आवास यक वाला` *(WER: 41.7%)*
- **KenLM Soft (α=0.15)**: `त्याला नदी पार करायचे होती तेवढया त्याला घोड्यांच्या टापांचा आवास यक वाला` *(WER: 50.0%)*
- **SraVaani 1.0**: `त्याला नदी पार करायची होती तेवढ्यात त्याला घोड्यांच्या टापांचा आवाज ऐकू आला` *(WER: 16.7%)*
- **Indic Conformer**: `त्याला नदी पार करायची होती तेवढ्यात त्याला घोडयाच्या टापांचा आवाज ऐकू आला` *(WER: 16.7%)*
- **Failure Causes**: Phonetic Substitution / Unseen Vocabulary
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

### Example M-4 (`sample_00009`)
- **Reference**: `मान खाली घालुन तसाच पुढे गेलो, काय आश्चर्य स्वर्गात मला औरंगजेब भेटला, म्हणे काफरांना मारुनच तर स्वर्ग गाठला.`
- **Greedy CTC**: `मानखाली घालून तसाच पुढे गेलो काय आश्चरय सर्गात मला अवरंगचे भेटला म्हणे कापराना मारूनच तरसवर्गगाठला` *(WER: 61.1%)*
- **Lexicon Beam**: `मानखाली घालून तसाच पुढे गेलो काय आश्चरय स्र्गात मला अवरंगचे भेटला म्हणे कापराना मारूनच तर सवर्गगाठला` *(WER: 55.6%)*
- **KenLM Default (α=0.5)**: `मान खाली घालून तसाच पुढे गेलो काय आश्रयसर्गात मला अवरंगचेभेटला म्हणे कारन मारून तर स्वर्गगाठला` *(WER: 50.0%)*
- **KenLM Soft (α=0.15)**: `मान खाली घालून तसाच पुढे गेलो काय आश्चर्य सर्गात मला अवरंगचे भेटला म्हणे कापा ना मारून तर स्वर्ग गाठला` *(WER: 33.3%)*
- **SraVaani 1.0**: `मान खाली घालून तसाच पुढे गेलो काय आश्चर्य स्वर्गात मला औरंगजेब भेटला म्हणे काफरांना मारूनच तर स्वर्ग गाठला` *(WER: 11.1%)*
- **Indic Conformer**: `मान खाली घालून तसाच पुढे गेलो काय आश्चर्य स्वर्गात मला औरंगजेब भेटला म्हणे काफरांना मारूनच तर स्वर्ग गाठला` *(WER: 11.1%)*
- **Failure Causes**: Postposition Compound/Spacing (विभक्ती प्रत्यय)
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

### Example M-5 (`sample_00011`)
- **Reference**: `भिल्ल वगैरे जातींची लहान लहान राज्ये होती.`
- **Greedy CTC**: `भिल्ल वगरे जातींची लहान लहान राज्य होते` *(WER: 42.9%)*
- **Lexicon Beam**: `भिल्ल वगरे जातींची लहान लहान राज्य होते` *(WER: 42.9%)*
- **KenLM Default (α=0.5)**: `भिन्न वगैरे जातींची लहान लहान राज्य होते` *(WER: 42.9%)*
- **KenLM Soft (α=0.15)**: `बिल्ल वगैरे जातींची लहान लहान राज्य होते` *(WER: 42.9%)*
- **SraVaani 1.0**: `भिल्लो वगैरे जातींची लहान लहान राज्ये होती` *(WER: 14.3%)*
- **Indic Conformer**: `भिल्ल वगैरे जातींची लहान लहान राज्ये होती` *(WER: 0.0%)*
- **Failure Causes**: Phonetic Substitution / Unseen Vocabulary
- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.

---

## 3. Hard & Failed Sentences (Severe Discrepancies: WER > 50%)

These sentences highlight the principal failure boundaries of zero-shot acoustic transfer and greedy CTC.

### Example H-1 (`sample_00000`)
- **Reference**: `म्हणुनच महाराच बिरुद मी मानान वागवल`
- **Greedy CTC**: `म्हणूनच महाराच बिरूध मी मानानं वाढवला` *(WER: 66.7%)*
- **Lexicon Beam**: `म्हणूनच महाराच बिरूध मी मानानं वाढवला` *(WER: 66.7%)*
- **KenLM Default (α=0.5)**: `म्हणूनच महाराचबिरूध मी मानानवाढवला` *(WER: 83.3%)*
- **KenLM Soft (α=0.15)**: `म्हणूनच महाराच बिरूद मी मानानं वाढवला` *(WER: 66.7%)*
- **SraVaani 1.0**: `म्हणूनच महाराचं विरोध मी मानानं वागवलं` *(WER: 83.3%)*
- **Indic Conformer**: `म्हणूनच महाराचं बिरूद मी मानानं वागवलं` *(WER: 83.3%)*
- **Root Cause Analysis**: **Phonetic Substitution / Unseen Vocabulary**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.

### Example H-2 (`sample_00001`)
- **Reference**: `स्वर्णाचारीने घाबरण्याचं नाटक करत म्हटलं, 'नायकजी, मला सिंहाचा आहार बनवण्यासाठी त्याच्यापुढे फेकून द्या!`
- **Greedy CTC**: `स्वरणा चारीने खाबरण्याचा नाटक करत म्हटल नायकजी मला सिवहाचा आहार बनवण्यासाठी त्याच्यापुढे फेकून द्या` *(WER: 38.5%)*
- **Lexicon Beam**: `स्वरणा चारीने खाबरण्याचा नाटक करत म्हटल नायकजी मला सिवहाचा आहार बनवण्यासाठी त्याच्यापुढे फेकून द्या` *(WER: 38.5%)*
- **KenLM Default (α=0.5)**: `स्वरणाचारीने घाबरण्याचा नाटक करत म्हटलं नायजी,मला सिंह चा आहार बनवण्यासाठी त्याच्या पुढे फेकून द्या` *(WER: 53.9%)*
- **KenLM Soft (α=0.15)**: `स्वरणा चारी ने खा भरण्याच नाटक करत म्हटल नाय जी मला सिंह चा आहार बनवण्यासाठी त्याच्या पुढे फेकून द्या` *(WER: 92.3%)*
- **SraVaani 1.0**: `स्वर्णाचारीने घाबरण्याचं नाटक करत म्हटलं नायकजी मला सिंहाचा आहार बनवण्यासाठी त्याच्या पुढे फेकून द्या` *(WER: 15.4%)*
- **Indic Conformer**: `स्वर्णाचारीने घाबरण्याचं नाटक करत म्हटलं नायकजी मला सिंहाचा आहार बनवण्यासाठी त्याच्यापुढे फेकून द्या` *(WER: 0.0%)*
- **Root Cause Analysis**: **Anusvara / Nasalization Variant**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.

### Example H-3 (`sample_00002`)
- **Reference**: `घोडयावरून खाली उतरताना घोडेस्वार वृध्दाला म्हणाला, बाबा एवढया कडाक्याच्या थंडीत नदी कडेला तुम्ही किती वेळ बसला होतात.`
- **Greedy CTC**: `मोड्यावरून खाली उतरताना घोडेश्वार ृरद्ाला म्हणाला बाबा एवढह्या कडाक्याच्या थंडीत नदीकडेला तुम्ही कितीवेळ बसला होतात` *(WER: 47.1%)*
- **Lexicon Beam**: `मोड्यावरून खाली उतरताना घोडेश्वार ृरद्ाला म्हणाला बाबा एवढह्या कडाक्याच्या थंडीत नदीकडेला तुम्ही कितीवेळ बसला होतात` *(WER: 47.1%)*
- **KenLM Default (α=0.5)**: `गोड्या वरून खाली उतरताना घोडे वाढ प्रदान म्हणाला बाबा एवढा कडा त्याच्या थंडीत नदी कडेला तुम्ही किती वेळ बसला होतात` *(WER: 47.1%)*
- **KenLM Soft (α=0.15)**: `गोड्या वरून खाली उतर ताना घोडे स्वार प्रदान म्हणाला बाबा एवढा कडा त्याच्या थंडीत नदी कडेला तुम्ही किती वेळ बसला होतात` *(WER: 58.8%)*
- **SraVaani 1.0**: `घोड्यावरून खाली उतरताना घोडेस्वार वृद्धाला म्हणाला बाबा एवढ्या कडाक्याच्या थंडीत नदीकडेला तुम्ही किती वेळ बसला होतात` *(WER: 29.4%)*
- **Indic Conformer**: `घोड्यावरून खाली उतरताना घोडेस्वार वृद्धधाला म्हणाला बाबा एवढ्या कडाक्याच्या थंडीत नदीकडेला तुम्ही किती वेळ बसला होतात` *(WER: 29.4%)*
- **Root Cause Analysis**: **Postposition Compound/Spacing (विभक्ती प्रत्यय)**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.

### Example H-4 (`sample_00004`)
- **Reference**: `त्याचाशी बोलत असतानाच अचानक खांद्यावर भक्कम अशी थाप पडली.`
- **Greedy CTC**: `त्याच्याशी बोलत असतानाच अजयानत खां्यावर भक्कम अशीत थाप पडले` *(WER: 55.6%)*
- **Lexicon Beam**: `त्याच्याशी बोलत असतानाच अज्यानत खां्यावर भक्कम अशीत थाप पडले` *(WER: 55.6%)*
- **KenLM Default (α=0.5)**: `त्याच्याशी बोलत असतानाच अचानक खाल्यावर भक्कम अशी खा पडले` *(WER: 44.4%)*
- **KenLM Soft (α=0.15)**: `त्याच्याशी बोलत असतानाच अजयानत खाल्यावर भक्कम अशी छाप पडले` *(WER: 55.6%)*
- **SraVaani 1.0**: `त्याच्याशी बोलत असतानाच अचानक खांद्यावर भक्कम अशी थाप पडली` *(WER: 11.1%)*
- **Indic Conformer**: `त्याच्याशी बोलत असतानाच अचानक खांद्यावर भक्कम अशी थाप पडली` *(WER: 11.1%)*
- **Root Cause Analysis**: **Phonetic Substitution / Unseen Vocabulary**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.

### Example H-5 (`sample_00006`)
- **Reference**: `त्याप्रमाणे ती मुले एकेक काठी घेऊन आली.`
- **Greedy CTC**: `त्याप्रमाणे तीमुले एथेक काठी घेवऊन ाले` *(WER: 71.4%)*
- **Lexicon Beam**: `त्याप्रमाणे तीमुले एथेक काठी घेवऊन ाले` *(WER: 71.4%)*
- **KenLM Default (α=0.5)**: `त्याप्रमाणे ती मुले एक काठी घेऊन आले` *(WER: 28.6%)*
- **KenLM Soft (α=0.15)**: `त्या प्रमाणे ती मुले एक काठी घेऊन आले` *(WER: 57.1%)*
- **SraVaani 1.0**: `त्याप्रमाणे ती मुले एक एक काठी घेऊन आली` *(WER: 28.6%)*
- **Indic Conformer**: `त्याप्रमाणे ती मुले एके एक काठी घेऊन आली` *(WER: 28.6%)*
- **Root Cause Analysis**: **Phonetic Substitution / Unseen Vocabulary**
- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.


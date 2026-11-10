## Statistik keyphrase (BART-KPG)

| Statistik | SciFact | NFCorpus |
|---|---|---|
| Rata-rata keyphrase per dokumen | 4.44 | 4.33 |
| Dokumen tanpa keyphrase | 0 | 0 |
| Proporsi keyphrase dokumen *absent* | 0.054 | 0.045 |
| Rata-rata keyphrase per query (test) | 2.92 | 1.89 |
| Query tanpa keyphrase | 0 | 0 |
| Proporsi keyphrase query *absent* | 0.147 | 0.285 |
| Query dengan skor keyphrase 0 untuk semua dokumen | 1 | 32 |
|   … sebagai proporsi query test | 0.003 | 0.099 |
| Median jumlah dokumen dengan skor keyphrase > 0 per query | 155 | 38 |
| Query dengan nDCG@10 Keyphrase-only = 0 | 132 | 133 |
|   … sebagai proporsi query test | 0.440 | 0.412 |
| Query dengan nDCG@10 Semantic-only = 0 (pembanding) | 60 | 100 |
| Rata-rata dokumen relevan per query test | 1.1 | 38.2 |
| Pasangan (query, dokumen relevan) dengan skor keyphrase > 0 | 0.805 | 0.140 |
| Query dengan ≥ 1 dokumen relevan berskor keyphrase > 0 | 245 | 230 |
|   … sebagai proporsi query test | 0.817 | 0.712 |

*Absent* = keyphrase yang setelah `text.normalize_phrase` (huruf kecil, stopword di tepi dibuang, stemming) tidak muncul sebagai urutan kata utuh di judul + abstrak (atau teks query) yang di-`tokenize` dan di-`stem` dengan aturan yang sama.

### Proporsi *absent* menurut urutan keluaran generator

| Urutan ke- | SciFact absent | SciFact n frasa | NFCorpus absent | NFCorpus n frasa |
|---|---|---|---|---|
| 1 | 0.033 | 5183 | 0.025 | 3633 |
| 2 | 0.033 | 5183 | 0.030 | 3633 |
| 3 | 0.052 | 5143 | 0.047 | 3581 |
| 4 | 0.086 | 4316 | 0.072 | 2790 |
| 5 | 0.076 | 2185 | 0.067 | 1307 |
| 6 | 0.087 | 704 | 0.060 | 447 |
| 7 | 0.105 | 190 | 0.074 | 176 |
| 8 | 0.030 | 66 | 0.024 | 84 |
| 9 | 0.222 | 27 | 0.038 | 53 |
| 10 | 0.118 | 17 | 0.036 | 28 |

Urutan akhir (9–10) hanya punya sedikit frasa karena kebanyakan dokumen mendapat < 9 keyphrase; proporsinya tidak stabil.

### Contoh keyphrase dokumen: SciFact (5 dokumen acak, seed 13)

- **[6517267] Barriers and facilitators to implement shared decision making in multidisciplinary sciatica care: a qualitative study**: barriers (1.0); facilitators (0.8); shared decision making (0.67); sciatica (0.57); qualitative study (0.5)
- **[8551160] Mitochondria: Dynamic Organelles in Disease, Aging, and Development**: mitochondria (1.0); disease (0.8); aging (0.67); development (0.57); fusion (0.5); fission (0.44)
- **[11244195] STAR: ultrafast universal RNA-seq aligner.**: rna-seq (1.0); alignment (0.8); splicing (0.67); canonical junctions (0.57); intergenic splices (0.5)
- **[12948892] Ca2+-permeable AMPA receptors regulate growth of human glioblastoma via Akt activation.**: ca2+-permeable ampa receptor (1.0); human glioblastoma (0.8); akt activation (0.67)
- **[14437255] Congruent Visual Speech Enhances Cortical Entrainment to Continuous Auditory Speech in Noise-Free Conditions.**: continuous speech (1.0); audiovisual speech (0.8); multisensory integration (0.67); electroencephalography (0.57); speech perception (0.5, absent)

### Contoh keyphrase dokumen: NFCorpus (5 dokumen acak, seed 13)

- **[MED-2056] Cow's Milk Allergy Is a Major Contributor in Recurrent Perianal Dermatitis of Infants**: cow's milk allergy (1.0); recurrent perianal dermatitis (0.8); infant (0.67)
- **[MED-2207] Growth inhibition and induction of apoptosis in NB4 promyelocytic leukemia cells by trypsin inhibitor from sweet potato storage roots.**: growth inhibition (1.0); apoptosis (0.8); NB4 promyelocytic leukemia cells (0.67); trypsin inhibitor (0.57)
- **[MED-4479] Long-term exposure to sodium nitrite and risk of esophageal carcinoma: a cohort study for 30 years.**: long-term exposure (1.0); sodium nitrite (0.8); esophageal carcinoma (0.67); occupational exposure (0.57)
- **[MED-4484] Salt, processed meat and the risk of cancer.**: salt (1.0); processed meat (0.8); cancer (0.67)
- **[MED-5031] Sleep Habits and Susceptibility to the Common Cold**: sleep (1.0); susceptibility (0.8); common cold (0.67); rhinovirus (0.57)

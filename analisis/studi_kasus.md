## Studi kasus

Dipilih otomatis dari `per_query_ndcg10.csv`: *Hybrid menolong* = selisih nDCG@10 Hybrid − Semantic terbesar; *Hybrid kalah dari BM25* = selisih Hybrid − BM25 terkecil (seri diputus menurut id query). ✓ = relevan menurut qrels (angka = tingkat relevansi), ✗ = tidak relevan.

### Hybrid menolong dibanding Semantic: SciFact query 1185

- **Query:** The US health care system can save up to $750 million if 7% of patients waiting for kidney transplants participate in the optimized national kidney paired donation program.
- **Keyphrase query:** kidney paired donation (1.0), optimized national kidney transplantation program (0.8)
- **Dokumen relevan:** 1; selisih nDCG@10 Hybrid − Semantic = +0.5000, Hybrid − BM25 = +0.0000

| Skenario | nDCG@10 | Relevan pertama | #1 | #2 | #3 |
|---|---|---|---|---|---|
| BM25 | 1.0000 | 1 | ✓1 Kidney paired donation and optimizing the use of live donor  | ✗ Successful three-way kidney paired donation with cross-count | ✗ Call to Develop a Standard Acquisition Charge Model for Kidn |
| Semantic | 0.5000 | 3 | ✗ Managing finances of shipping living donor kidneys for donor | ✗ Call to Develop a Standard Acquisition Charge Model for Kidn | ✓1 Kidney paired donation and optimizing the use of live donor  |
| Keyphrase | 0.6309 | 2 | ✗ Successful three-way kidney paired donation with cross-count | ✓1 Kidney paired donation and optimizing the use of live donor  | ✗ Call to Develop a Standard Acquisition Charge Model for Kidn |
| Hybrid | 1.0000 | 1 | ✓1 Kidney paired donation and optimizing the use of live donor  | ✗ Successful three-way kidney paired donation with cross-count | ✗ Call to Develop a Standard Acquisition Charge Model for Kidn |

**Dokumen relevan utama:** [16737210] Kidney paired donation and optimizing the use of live donor organs.  
Keyphrase dokumen: kidney paired donation, optimization, matching algorithms, end-stage renal disease, renal transplantation  
**Penyebab (dari angka):** Dokumen relevan [16737210] ada di peringkat #3 pada Semantic-only dan naik ke #1 pada Hybrid. Unit keyphrase yang sama dengan query: p:kidney pair donat, t:donat, t:kidney, t:optim, t:pair, t:transplant (s_kp = 0.667); skor semantiknya 0.592. Sinyal keyphrase menambah skor dokumen ini setelah normalisasi min-max, sehingga peringkatnya naik.

### Hybrid kalah dari BM25: SciFact query 1024

- **Query:** Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes.
- **Keyphrase query:** recurrent mutations (1.0), ctcf anchor sites (0.8), oncogenes (0.67)
- **Dokumen relevan:** 1; selisih nDCG@10 Hybrid − Semantic = +0.0000, Hybrid − BM25 = -1.0000

| Skenario | nDCG@10 | Relevan pertama | #1 | #2 | #3 |
|---|---|---|---|---|---|
| BM25 | 1.0000 | 1 | ✓1 3D Chromosome Regulatory Landscape of Human Pluripotent Cell | ✗ Specific sites in the C terminus of CTCF interact with the S | ✗ Characterization of constitutive CTCF/cohesin loci: a possib |
| Semantic | 0.0000 | 41 | ✗ Characterization of constitutive CTCF/cohesin loci: a possib | ✗ Genome architecture, rearrangements and genomic disorders. | ✗ TRF2 Recruits RTEL1 to Telomeres in S Phase to Promote T-Loo |
| Keyphrase | 0.0000 | >100 | ✗ Reflecting on 25 years with MYC | ✗ MicroRNAs: new tools for diagnosis, prognosis, and therapy i | ✗ Detecting and targeting tumor relapse by its resistance to i |
| Hybrid | 0.0000 | 49 | ✗ Characterization of constitutive CTCF/cohesin loci: a possib | ✗ Reflecting on 25 years with MYC | ✗ Colorectal Cancers from Distinct Ancestral Populations Show  |

**Dokumen relevan utama:** [5373138] 3D Chromosome Regulatory Landscape of Human Pluripotent Cells.  
Keyphrase dokumen: pluripotent cells, transcriptional enhancer, insulator, chia-pet, chromosome structure, gene regulation  
**Penyebab (dari angka):** Dokumen relevan [5373138] ada di peringkat #1 pada BM25 tetapi #49 pada Hybrid (#41 pada Semantic-only). Term query yang cocok persis dengan dokumen: anchor, ctcf, frequent, mutat, occur, site. Unit keyphrase bersama: tidak ada (s_kp = 0.000). BM25 memberi bobot tinggi pada kecocokan kata yang langka, sedangkan embedding (s_sem = 0.434, peringkat 41 di seluruh korpus) dan keyphrase tidak menangkapnya cukup kuat.

### Hybrid menolong dibanding Semantic: NFCorpus query PLAIN-681

- **Query:** betel nuts
- **Keyphrase query:** betel nuts (1.0)
- **Dokumen relevan:** 1; selisih nDCG@10 Hybrid − Semantic = +0.3691, Hybrid − BM25 = +0.0000

| Skenario | nDCG@10 | Relevan pertama | #1 | #2 | #3 |
|---|---|---|---|---|---|
| BM25 | 1.0000 | 1 | ✓1 Association between betel-nut chewing and chronic kidney dis | ✗ Diet and prevention of oral cancer: strategies for clinical  | ✗ Nut consumption, vegetarian diets, ischemic heart disease ri |
| Semantic | 0.6309 | 2 | ✗ The role of nuts in the optimal diet: time for a critical ap | ✓1 Association between betel-nut chewing and chronic kidney dis | ✗ Nuts: anti-atherogenic food? |
| Keyphrase | 1.0000 | 1 | ✓1 Association between betel-nut chewing and chronic kidney dis | ✗ Diet and prevention of oral cancer: strategies for clinical  | ✗ A review of the evidence: nuts and body weight. |
| Hybrid | 1.0000 | 1 | ✓1 Association between betel-nut chewing and chronic kidney dis | ✗ The role of nuts in the optimal diet: time for a critical ap | ✗ Nuts: anti-atherogenic food? |

**Dokumen relevan utama:** [MED-5017] Association between betel-nut chewing and chronic kidney disease in men.  
Keyphrase dokumen: betel-nut chewing, chronic kidney disease, men  
**Penyebab (dari angka):** Dokumen relevan [MED-5017] ada di peringkat #2 pada Semantic-only dan naik ke #1 pada Hybrid. Unit keyphrase yang sama dengan query: t:betel, t:nut (s_kp = 0.399); skor semantiknya 0.370. Sinyal keyphrase menambah skor dokumen ini setelah normalisasi min-max, sehingga peringkatnya naik.

### Hybrid kalah dari BM25: NFCorpus query PLAIN-1473

- **Query:** kohlrabi
- **Keyphrase query:** kohlrabi (1.0)
- **Dokumen relevan:** 1; selisih nDCG@10 Hybrid − Semantic = +0.0000, Hybrid − BM25 = -1.0000

| Skenario | nDCG@10 | Relevan pertama | #1 | #2 | #3 |
|---|---|---|---|---|---|
| BM25 | 1.0000 | 1 | ✓1 Glucoraphanin and 4-hydroxyglucobrassicin contents in seeds  | - | - |
| Semantic | 0.0000 | >100 | ✗ Hibiscus anthocyanins rich extract-induced apoptotic cell de | ✗ Variability of iodine content in common commercially availab | ✗ Marine edible algae as disease preventers. |
| Keyphrase | 0.0000 | >100 | - | - | - |
| Hybrid | 0.0000 | >100 | ✗ Hibiscus anthocyanins rich extract-induced apoptotic cell de | ✗ Variability of iodine content in common commercially availab | ✗ Marine edible algae as disease preventers. |

**Dokumen relevan utama:** [MED-4455] Glucoraphanin and 4-hydroxyglucobrassicin contents in seeds of 59 cultivars of broccoli, raab, kohlrabi, radish, cauliflower, brussels sprouts, kal...  
Keyphrase dokumen: glucoraphanin, 4-hydroxyglucobrassicin, seeds  
**Penyebab (dari angka):** Dokumen relevan [MED-4455] ada di peringkat #1 pada BM25 tetapi di luar 100 besar pada Hybrid (di luar 100 besar pada Semantic-only). Term query yang cocok persis dengan dokumen: kohlrabi. Unit keyphrase bersama: tidak ada (s_kp = 0.000). BM25 memberi bobot tinggi pada kecocokan kata yang langka, sedangkan embedding (s_sem = 0.112, peringkat 119 di seluruh korpus) dan keyphrase tidak menangkapnya cukup kuat.

### Kandidat lain

| Jenis | Dataset | Query | Hybrid − Semantic | Hybrid − BM25 | Teks query |
|---|---|---|---|---|---|
| Hybrid menolong dibanding Semantic | SciFact | 295 | +0.5000 | +0.0000 | Crosstalk between dendritic cells (DCs) and innate lymphoid cells (ILCs) is impo |
| Hybrid menolong dibanding Semantic | SciFact | 527 | +0.3869 | -0.2441 | Homozygous deletion of murine Sbds gene from osterix-expressing mesenchymal stem |
| Hybrid kalah dari BM25 | SciFact | 1337 | -0.2891 | -1.0000 | Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164. |
| Hybrid kalah dari BM25 | SciFact | 1363 | +0.0000 | -1.0000 | Venules have a thinner or absent smooth layer compared to arterioles. |
| Hybrid menolong dibanding Semantic | NFCorpus | PLAIN-383 | +0.2080 | +0.1696 | What do you think of Dr. Jenkins' take on paleolithic diets? |
| Hybrid menolong dibanding Semantic | NFCorpus | PLAIN-3191 | +0.1602 | +0.0626 | Is Distilled Fish Oil Toxin-Free? |
| Hybrid kalah dari BM25 | NFCorpus | PLAIN-2081 | +0.0000 | -1.0000 | shelf life |
| Hybrid kalah dari BM25 | NFCorpus | PLAIN-1288 | +0.0000 | -0.7753 | grapes |

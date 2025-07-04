# COLON-AID
This repository provides the implementation and demonstration of the COLON-AID model, prediciting survival risks based 
on real-world medical records.

# System requirement
- Ubuntu 20.04 or higher
- Python 3.10 or higher

# Installation
```bash
pip install -r requirements.txt
```
Typical installation time: within 30 minutes, depending on the network speed (e.g., 10MB/s). 
If you have cached `pytorch` package, the installation time will be significantly reduced to within 10 minutes.

# Usage
```bash
python predict_surv.py data/samples/1.json
```

## Expected output
```
Patient records: [
    {
        "date": "2010-06-26",
        "event_type": "出入院记录",
        "event_type_en": "admission",
        "content": "入院日期：2010-06-26\n性别：男\n初次病理诊断年龄：83\n种族：白人\n族裔：非西班牙裔或拉丁裔\n提交的肿瘤部位：结肠\n其他恶性肿瘤史：无\n新辅助治疗史：无\n初次病理诊断年份：2010\n解剖器官子分部：升结肠\n诊断时体重（kg）：75\n诊断时身高（cm）：168\n结肠息肉史：无\n获取时结肠息肉指标：是\n结直肠癌家族史：0",
        "content_en": "Admission Date: 2010-06-26\nGender: MALE\nAge at Initial Pathologic Diagnosis: 83\nRace: WHITE\nEthnicity: NOT HISPANIC OR LATINO\nSubmitted Tumor Site: Colon\nHistory of Other Malignancy: NO\nHistory of Neoadjuvant Treatment: No\nYear of Initial Pathologic Diagnosis: 2010\nAnatomic Organ Subdivision: Ascending Colon\nWeight at Diagnosis (kg): 75\nHeight at Diagnosis (cm): 168\nHistory of Colon Polyps: NO\nColon Polyps at Procurement Indicator: YES\nFamily History of Colorectal Cancer: 0\n"
    },
    {
        "date": "2010-07-10",
        "event_type": "病理报告",
        "event_type_en": "pathology_report",
        "content_en": "Here's a detailed summary of the pathological report:\nClinical Information:\n- Diagnosis: Cancer of the large intestine\nExamination Details:\n- Specimen:\n- 19.9 cm segment of the large intestine with peri-intestinal tissue (25 x 13 x 3 cm)\n- 8 cm segment of the small intestine\n- 7 cm appendix\nMacroscopic Description:\n- Tumor:\n- Cauliflower-shaped, located in the intestinal mucosa\n- Size: 4.7 x 9.3 x 1.2 cm\n- Involves 100% of the intestine's circumference\n- Proximity: 9.2 cm from the proximal cut end, 12.2 cm from the distal cut end, and 1.2 cm from the ileocecal valve\n- Additional Findings:\n- Polyps up to 1.5 cm in size\nMicroscopic Description:\n- Tumor Type:\n- Tubulopapillary adenocarcinoma, partially mucinous (G3 grade)\n- Deep infiltration into the muscular layer and pericolonic adipose tissue\n- Other Findings:\n- Intestine ends free of neoplastic lesions\n- Tubulopapillary adenomas with moderate to severe dysplasia\n- Reactive lymphonodulitis (No VII)\n- Appendix without lesions\nHistopathological Diagnosis:\n- Primary Diagnosis: Tubulopapillary adenocarcinoma, partially mucinous, of the colon (G3; Dukes B; Astler-Coller B2; pT3; pNO)\n- Secondary Findings: Tubulopapillary adenomas with high-grade dysplasia",
        "content": "以下是病理报告的详细摘要：\n临床信息：\n- 诊断：结肠癌\n检查细节：\n- 标本：\n- 大肠段（25 x 13 x 3 cm）及周围组织，长度为19.9厘米\n- 小肠段，长度为8厘米\n- 阑尾，长度为7厘米\n宏观描述：\n- 肿瘤：\n-厘米，距离回盲瓣1.2厘米\n- 附加发现：\n- 多发性息肉，最大直径1.5厘米\n微观描述：\n- 肿瘤类型：\n- 管状乳头状腺癌，部分为黏液性（G3级）\n- 深度浸润至肌层和周围脂肪组织\n- 其他发现：\n- 肠段两端无肿瘤病变\n- 管状乳头状腺瘤r-Coller B2；pT3；pNO）\n- 次要发现：管状乳头状腺瘤，高度不典型增生"
    }
]
Predicted risk:  5.10
Risk Percentile:  50.80%
```

## Expected run time
The expected run time for the prediction is within 5 seconds.


# More samples
4 samples are provided in the `data/samples` directory. You can run the prediction on these samples using the same command as above, replacing the file name with the desired sample.

#prompting GPT-4 with following category names
srh_cat = { 'hgg' : 'high grade glioma', 'lgg' : 'low grade glioma', 'menin' : 'meningioma', 'pit' : 'pituitary adenoma', 
           'sch' : 'schwannoma', 'met' : 'metastatic tumor', 'normal' : 'normal brain tissue'}

#prompt1 : Give 4 different descriptions for the phrase: "normal brain tissue"?

srh_desc1 = {

'hgg' :  ["High grade glioma refers to an aggressive type of brain tumor originating in the glial cells of the brain or spinal cord. These tumors are characterized by their rapid growth and typically exhibit malignant behavior.",
"A high grade glioma is a type of primary brain tumor characterized by its advanced stage of malignancy. It arises from the supportive glial cells of the nervous system and is known for its invasive nature and potential to spread within the brain.",
"High grade glioma is a term used to describe a group of fast-growing and highly malignant brain tumors that develop from glial cells. These tumors are often associated with poor prognosis due to their aggressive nature and resistance to treatment.",
"In medical parlance, high grade glioma denotes a category of brain tumors that are of a high histological grade, implying aggressive cellular features and rapid proliferation. These tumors are often challenging to treat and pose a significant threat to the patient's neurological health."],

'lgg' : ["Low grade glioma is a term used to describe a less aggressive type of brain tumor that originates from glial cells in the brain or spinal cord. These tumors are characterized by their slow growth and a generally more favorable prognosis compared to high grade gliomas.",
"A low grade glioma refers to a primary brain tumor that arises from glial cells and exhibits a relatively lower degree of malignancy. These tumors tend to grow slowly and are typically associated with better treatment outcomes.",
"Low grade glioma is a category of brain tumors characterized by their less aggressive nature, typically originating from glial cells. These tumors progress slowly and are generally associated with a more favorable long-term prognosis compared to their high grade counterparts.",
"In medical terminology, low grade glioma denotes a group of brain tumors that are histologically less malignant and exhibit a slower growth rate. These tumors, originating from glial cells, are generally more amenable to treatment, and patients often have a better quality of life compared to those with high grade gliomas."],

'menin' : ["A meningioma is a typically benign tumor that originates from the meninges, the protective membranes surrounding the brain and spinal cord. These tumors are slow-growing and often non-cancerous, but their location can cause symptoms depending on their size and position.",
"Meningioma is a type of brain tumor that develops from the meninges, the thin layers of tissue that cover and protect the brain and spinal cord. Most meningiomas are non-cancerous and tend to grow slowly, although their symptoms can vary based on their location.",
"A meningioma is a tumor that forms in the meninges, which are the layers of tissue that envelope the brain and spinal cord. These tumors are typically non-malignant and exhibit slow growth, but their size and location can lead to neurological symptoms depending on their impact on nearby structures.",
"Meningioma is a type of intracranial tumor that arises from the meninges, the membranes that encase the central nervous system. While most meningiomas are benign, they can cause symptoms such as headaches or neurological deficits when they press on surrounding brain tissue due to their slow, space-occupying growth."],

'pit' : ["A pituitary adenoma is a non-cancerous tumor that develops in the pituitary gland, a small gland located at the base of the brain. These tumors can affect hormone production and lead to various hormonal imbalances.",
"Pituitary adenoma refers to a benign growth or tumor that occurs in the pituitary gland, a critical endocrine organ situated at the base of the brain. These tumors can disrupt the gland's normal function and result in hormonal disturbances.",
"A pituitary adenoma is a non-malignant tumor that forms within the pituitary gland, which regulates hormone production. These tumors can vary in size and affect hormone levels, leading to a range of symptoms and health issues.",
"Pituitary adenoma is a term used to describe a typically benign tumor that originates in the pituitary gland, an essential gland responsible for regulating various hormones in the body. These tumors can disrupt hormonal balance and cause a wide range of health problems depending on their size and type."],

'sch' : ["A schwannoma, also known as a neurilemmoma, is a benign tumor that originates from Schwann cells, which are responsible for forming the insulating myelin sheath around nerve fibers. These tumors most commonly develop in peripheral nerves, such as those in the head, neck, or extremities.",
"Schwannoma is a non-cancerous tumor that arises from Schwann cells, which are specialized cells that wrap around and insulate peripheral nerves. These tumors can grow in various locations throughout the body and are often slow-growing and encapsulated.",
"A schwannoma is a type of benign nerve sheath tumor that arises from Schwann cells. These tumors are usually solitary, encapsulated growths that can develop along peripheral nerves, potentially causing neurological symptoms depending on their location and size.",
"Schwannoma, also called neurilemmoma, is a non-malignant tumor originating from Schwann cells in peripheral nerves. These tumors are typically well-defined, slow-growing, and can occur in different parts of the body, including the head, neck, and extremities."],

'met' : ["A metastatic tumor, also known as a secondary tumor, is a cancerous growth that has spread from its original site (primary tumor) to other parts of the body through the bloodstream or lymphatic system. These tumors are formed by cancer cells that have detached from the primary tumor and migrated to distant organs or tissues.",
"Metastatic tumor refers to a cancerous lesion that has developed in a location away from the primary tumor site. These tumors are the result of cancer cells breaking away from the primary tumor, traveling through the bloodstream or lymphatic system, and establishing new cancer growths elsewhere in the body.",
"A metastatic tumor is a cancerous growth that has originated from cells of a primary tumor but has spread to different sites within the body. This process, known as metastasis, can occur when cancer cells travel through the circulatory or lymphatic systems and form new tumors in distant organs or tissues.",
"Metastatic tumor, or metastasis, occurs when cancer cells from a primary tumor migrate to other parts of the body and form new cancerous growths. These secondary tumors can develop in various organs or tissues, often leading to more advanced stages of cancer with different clinical manifestations."],

'normal' : ["Normal brain tissue refers to the healthy and unaltered cellular and structural components of the brain. It encompasses the neurons, glial cells, blood vessels, and supporting structures that make up the brain in its natural state.",
"Normal brain tissue is the standard and unaffected composition of the brain, characterized by its proper functioning neurons, glial cells, and absence of abnormalities or disease. It plays a crucial role in maintaining cognitive and neurological functions.",
"Normal brain tissue refers to the undamaged and healthy cells and structures that compose the brain. It is vital for the brain's normal functioning, including processes such as cognition, sensory perception, and motor control.",
"Normal brain tissue is the baseline state of the brain without any pathological changes or abnormalities. It consists of well-organized neurons, supportive glial cells, and a network of blood vessels, contributing to the brain's essential functions and overall health."]}





#prompt2 : List visual objects or characteristics usually seen with the tumor category: 'high grade glioma'?

srh_desc2 = {

'hgg' :  ['Contrast-enhancing regions on brain imaging (e.g., MRI) due to increased blood flow and leakage of the blood-brain barrier.',
'Irregular and invasive growth patterns, often with poorly defined borders.',
'Presence of necrotic (dead) tissue within the tumor.',
'Mass effect, which can lead to compression and displacement of surrounding brain structures.',
'Edema or swelling in the adjacent brain tissue.',
'Increased intracranial pressure, which can result in symptoms such as headaches and vomiting.',
'Neurological deficits, including motor or sensory impairments, depending on the tumor location.',
'In some cases, the tumor may cause seizures or changes in cognitive function.'],

'lgg' : ['Low-grade gliomas often have well-defined borders, making them distinguishable from the surrounding brain tissue on imaging.',
'These tumors are typically solid in nature and do not exhibit the central necrosis commonly seen in high-grade gliomas.',
'Low-grade gliomas may exhibit minimal or no contrast enhancement on imaging studies like MRI, indicating a lower degree of vascularity compared to high-grade gliomas.',
'They tend to grow slowly over time, which can be observed in serial imaging studies.',
'Low-grade gliomas are less likely to cause significant mass effect, meaning they are less likely to compress or displace adjacent brain structures.',
'Patients with low-grade gliomas are less likely to experience symptoms related to increased intracranial pressure, such as severe headaches or vomiting.',
'These tumors often have a more extended clinical course compared to high-grade gliomas, with patients experiencing symptoms over a more extended period.',
'In some cases, low-grade gliomas may contain areas of calcification, which can sometimes be seen on imaging.'],

'menin' : ['Meningiomas are often found outside of the brain tissue, originating from the meninges, the protective membranes surrounding the brain and spinal cord.',
'They tend to have well-defined, distinct borders on imaging, making them stand out from the surrounding brain tissue.',
'Meningiomas usually display a uniform and strong contrast enhancement on imaging studies like MRI, indicating a rich blood supply.',
'In some cases, they may exhibit a "dural tail" sign, where they appear to extend a short distance into the adjacent dura mater, the outermost meningeal layer.',
'These tumors typically grow slowly over time, and their slow growth rate can be observed through serial imaging studies.',
'Some meningiomas may contain areas of calcification, which can be visible on imaging as regions of increased density.',
'Meningiomas can occur at various locations along the meninges, including the convexity of the brain, parasagittal region, skull base, and the falx cerebri.',
'Depending on their size and location, meningiomas can exert mass effect on adjacent brain structures, leading to symptoms such as headaches, neurological deficits, or seizures.',
'The enhancement pattern of meningiomas on contrast-enhanced MRI can vary, with some showing intense and homogeneous enhancement while others may exhibit more heterogeneous patterns.',
'While the majority of meningiomas are benign (non-cancerous), their growth and location can lead to symptoms and require medical intervention.'],

'pit' : ['Pituitary adenomas are typically located within the sella turcica, a bony cavity at the base of the brain.',
'They often exhibit well-defined borders on imaging studies, making them distinguishable from the surrounding brain tissue.',
'Pituitary adenomas can vary in size, ranging from small microadenomas to larger macroadenomas.',
'On T1-weighted MRI images, pituitary adenomas usually appear as hypointense (dark) lesions.'],

'sch' : ['Schwannomas often appear as well-defined masses with a homogeneous appearance on imaging studies, such as MRI.',
'These tumors can exhibit contrast enhancement on MRI scans, highlighting their vascularization.',
'Schwannomas may have a characteristic tapered or "dumbbell" shape, particularly when originating from nerve roots near the spinal cord.',
'They typically originate from peripheral nerves, such as cranial nerves or nerves in the extremities.',
'In some cases, schwannomas may show a peripheral ring-like pattern of enhancement on MRI scans.',
'They can vary in size, with some being small and slow-growing while others are larger.',
'Most schwannomas are benign (non-cancerous) tumors.',
'Depending on their location, they can compress or displace nearby nerves, blood vessels, or other structures, leading to symptoms.',
'Under a microscope, schwannomas often display a characteristic pattern of alternating Antoni A (cellular) and Antoni B (less cellular) areas, aiding in diagnosis.'],

'met' : ['Metastatic tumors are often identified as multiple discrete masses or nodules on imaging studies, suggesting their spread to different parts of the body.',
'They tend to appear in organs or tissues distant from the site of the primary cancer, such as the lungs, liver, bones, or brain.',
'Metastatic tumors frequently exhibit irregular shapes and borders on imaging, indicating their invasive nature and disruption of normal tissue architecture.',
'Contrast-enhanced imaging, such as contrast-enhanced CT or MRI scans, may reveal these tumors as areas with increased contrast uptake, distinguishing them from the surrounding tissue.',
'Depending on their location, metastatic tumors can lead to specific organ-related symptoms and functional impairments, such as breathing difficulties with lung metastases or neurological deficits with brain metastases.',
'The number and distribution of metastatic lesions can vary, ranging from a few isolated metastases to widespread involvement of multiple organs or regions.',
'In some cases, metastatic tumors may cause structural changes in the affected organs, such as bone destruction in the case of bone metastases.',
'Metastatic tumors often originate from primary cancers, and their identification aids in staging and determining the extent of cancer spread.',
'The characteristics of metastatic tumors may differ depending on the primary cancer type, and the imaging findings can help guide treatment decisions and prognosis assess'],

'normal' : ['Normal brain tissue appears on imaging studies as a uniform and symmetrical gray and white matter structure.',
'It is characterized by a lack of any distinct masses, nodules, or abnormalities within the brain parenchyma.',
'Normal brain tissue typically displays smooth and continuous contours on imaging, without irregular borders or disruptions.',
'On MRI or CT scans, normal brain tissue has a characteristic appearance, with gray matter appearing darker and white matter appearing lighter.',
'There are no areas of contrast enhancement within normal brain tissue, as contrast uptake usually indicates the presence of tumors or other pathological conditions.',
'Normal brain tissue exhibits a symmetrical distribution in both hemispheres of the brain, with similar appearance and characteristics on both sides.',
'It is generally free from structural deformities, bleeding, or fluid collections that might be present in abnormal brain tissue.',
'Normal brain tissue plays a vital role in various neurological functions and does not cause neurological deficits or symptoms when observed on imaging.']}


# key visual attributes visible in SRH data that doctors use to decide

def srh_desc3():
    
    srh_desc3 = {
  "visual_attributes": [
    {
      "attribute_name": "Cellularity",
      "description": "The density of cells within the tissue sample, which can vary between different tumor types and grades."
    },
    {
      "attribute_name": "Nuclear Pleomorphism",
      "description": "Variability in the size and shape of cell nuclei, which can be an indicator of tumor aggressiveness."
    },
    {
      "attribute_name": "Vascularity",
      "description": "The presence and pattern of blood vessels within the tumor, which can affect its growth and treatment options."
    },
    {
      "attribute_name": "Tissue Architecture",
      "description": "The arrangement and organization of cells and structures within the tissue, which can provide insights into the tumor's type and grade."
    },
    {
      "attribute_name": "Necrosis",
      "description": "The presence of dead or necrotic tissue within the tumor, which is often associated with high-grade tumors."
    },
    {
      "attribute_name": "Cell Infiltration",
      "description": "The extent to which tumor cells invade healthy brain tissue, which is crucial for treatment planning."
    },
    {
      "attribute_name": "Inflammatory Response",
      "description": "The presence of inflammatory cells or immune system response in the tissue, which can impact the tumor's behavior."
    },
    {
      "attribute_name": "Molecular Markers",
      "description": "The presence of specific molecular markers or genetic alterations, which can guide treatment choices."
    },
    {
      "attribute_name": "Lipid Content",
      "description": "Lipid content refers to the presence and distribution of lipids (fats) within brain tissue. Doctors are concerned about this attribute because certain brain tumor types may exhibit distinct lipid profiles. Lipid-rich regions in SRH images can indicate specific tumor characteristics and assist in classification."
    },
    {
      "attribute_name": "Protein Density",
      "description": "Protein density is a measure of the concentration and distribution of proteins in brain tissue. Doctors analyze this attribute because variations in protein density can provide insights into tumor growth and aggressiveness, as well as tissue health and integrity."
    },
    {
      "attribute_name": "Nucleic Acid Presence",
      "description": "Nucleic acid presence refers to the detection and localization of DNA and RNA within the tissue. Doctors are concerned about this attribute because the presence of nucleic acids can indicate areas of active cell proliferation or genetic mutations often seen in tumors."
    },
    {
      "attribute_name": "Cellular Morphology",
      "description": "Cellular morphology in SRH images reveals information about the size, shape, and organization of cells within the tissue. Doctors pay attention to this attribute because changes in cellular morphology can be indicative of tumor invasiveness, malignancy, and cellular atypia."
    },
    {
      "attribute_name": "Chemical Composition",
      "description": "Chemical composition indicates the distribution of different chemical components within the tissue, including specific molecules. Doctors are concerned about this attribute because it helps identify biomarkers, such as mutated proteins or metabolites, which can aid in tumor classification and treatment decisions."
    },
    {
      "attribute_name": "Microcalcifications",
      "description": "Microcalcifications are tiny calcium deposits that can be observed in certain brain tumors. They may present as small, bright specks in SRH images, and their presence can be indicative of specific tumor types."
    },
    {
      "attribute_name": "Cytoplasmic Inclusions",
      "description": "Cytoplasmic inclusions are abnormal structures within the cell's cytoplasm that may be visible in SRH images. They can provide insights into the type of brain tumor and its aggressiveness."
    },
    {
      "attribute_name": "Stromal Components",
      "description": "Stromal components refer to the non-cellular elements within the tumor's microenvironment, such as collagen fibers or fibrous tissue. The presence and organization of stromal components can vary and influence tumor behavior."
    },
    {
      "attribute_name": "Mitotic Figures",
      "description": "Mitotic figures are cells in the process of cell division and can be seen in SRH images. The frequency of mitotic figures can be an indicator of a tumor's growth rate and malignancy."
    },
    {
      "attribute_name": "Myelin Content",
      "description": "Myelin content pertains to the presence of myelin sheaths in the tissue. Variations in myelin content can be observed in different brain tumor types and can influence the appearance of SRH images."
    },
    {
      "attribute_name": "Nuclear-Cytoplasmic Ratio",
      "description": "The nuclear-cytoplasmic ratio refers to the proportion of the cell occupied by the nucleus. A higher ratio may indicate increased cellularity and is a concern for tumor analysis."
    },
    {
      "attribute_name": "Stem Cell Markers",
      "description": "Stem cell markers can be visualized in SRH images and are of interest to doctors as they may indicate the presence of cancer stem cells, which are associated with tumor growth and resistance to treatment."
    }
  ]
}
    
    return srh_desc3

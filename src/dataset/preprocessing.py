import numpy as np
import pandas as pd

class Preprocess:
    COLUMNS_TO_DROP = [
    # Constant or near-empty (carry no information)
    "Study ID", "Cancer Type", "Informed consent verified", "Is FFPE",
    "Somatic Status", "Tumor Disease Anatomic Site", "Disease code", "Project code",
    "Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value",
    # Redundant IDs / admin metadata
    "Other Patient ID", "Other Sample ID", "Pathology Report File Name",
    "Form completion date", "Tissue Source Site", "Oct embedded",
    "Tissue Prospective Collection Indicator", "Tissue Retrospective Collection Indicator",
    "Number of Samples Per Patient", "Days to Sample Collection.",
    # Leak the label (survival / follow-up outcomes)
    "Death from Initial Pathologic Diagnosis Date",
    "Days to Last Followup", "Disease Free (Months)", "Disease Free Status",
    "Person Neoplasm Status", "New Neoplasm Event Post Initial Therapy Indicator",
    "Metastatic tumor indicator", "Metastatic Site", "Metastatic Site Other",
    # Demographics
    "Diagnosis Age", "Birth from Initial Pathologic Diagnosis Date", "Sex",
    "Race Category", "Ethnicity Category", "Menopause Status", "Year Cancer Initial Diagnosis",
    "Prior Cancer Diagnosis Occurence",
    # Staging / histology / coding
    "American Joint Committee on Cancer Metastasis Stage Code",
    "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code",
    "Neoplasm Disease Stage American Joint Committee on Cancer Code",
    "American Joint Committee on Cancer Publication Version Type",
    "American Joint Committee on Cancer Tumor Stage Code",
    "Cancer Type Detailed", "Oncotree Code", "Neoplasm Histologic Type Name",
    "Tumor Other Histologic Subtype", "ICD-10 Classification",
    "International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code",
    "International Classification of Diseases for Oncology, Third Edition ICD-O-3 Site Code",
    "Primary Tumor Site", "Staging System", "Staging System.1",  # pandas renames the duplicate header
    # Lymph nodes
    "Primary Lymph Node Presentation Assessment Ind-3",
    "Positive Finding Lymph Node Hematoxylin and Eosin Staining Microscopy Count",
    "Positive Finding Lymph Node Keratin Immunohistochemistry Staining Method Count",
    "Lymph Node(s) Examined Number", "Micromet detection by ihc",
    # Receptor IHC / FISH (ER, PR, HER2)
    "ER positivity scale other", "ER positivity scale used", "ER Status By IHC",
    "ER Status IHC Percent Positive", "PR positivity define method",
    "PR positivity ihc intensity score", "PR positivity scale other", "PR positivity scale used",
    "PR status by ihc", "PR status ihc percent positive",
    "Cent17 Copy Number", "HER2 and cent17 cells count", "HER2 and cent17 scale other",
    "HER2 cent17 ratio", "HER2 copy number", "HER2 fish method", "HER2 fish status",
    "HER2 ihc percent positive", "HER2 ihc score", "HER2 positivity method text",
    "HER2 positivity scale other", "IHC-HER2", "IHC Score",
    "Brachytherapy first reference point administered total dose",  # actually mislabeled IHC scores
    # New-tumor-event (NTE) receptor fields (≤10 non-null each)
    "Nte cent 17 HER2 ratio", "Nte er ihc intensity score", "Nte er status",
    "Nte er status ihc positive", "Nte HER2 fish status", "Nte HER2 positivity ihc score",
    "Nte HER2 status", "Nte HER2 status ihc positive", "Nte pr ihc intensity score",
    "Nte pr status by ihc", "Nte pr status ihc positive",
    # Treatment / surgery
    "Neoadjuvant Therapy Type Administered Prior To Resection Text",
    "Adjuvant Postoperative Pharmaceutical Therapy Administered Indicator",
    "Did patient start adjuvant postoperative radiotherapy?",
    "First surgical procedure other", "Surgical procedure first",
    "Surgery for positive margins", "Surgery for positive margins other",
    "Margin status reexcision", "Disease Surgical Margin Status",
    "First Pathologic Diagnosis Biospecimen Acquisition Method Type",
    "First Pathologic Diagnosis Biospecimen Acquisition Other Method Type",
    # Genomic summaries (not expression)
    "Fraction Genome Altered", "Mutation Count", "TMB (nonsynonymous)",
    ]
    def __init__(self, cga_dataset : str, mrna_dataset : str) -> None:
        self.cga_dataset = cga_dataset
        self.mrna_dataset = mrna_dataset
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Returns one row per primary-tumor sample: Patient ID, label, Overall Survival (Months), then log2(RSEM + 1) genes.
        """
        df_cga = self.TextToCsv(file_path=self.cga_dataset)
        df_cga = self.drop_columns(df_cga=df_cga).set_index("Sample ID")
        df_mrna = self.load_mrna()
        df_mrna = df_mrna.loc[df_mrna.index.intersection(df_cga.index)]  # filter genes on the samples we train on
        df_mrna = self.eliminate_zero_genes(df_mrna=df_mrna)
        df_mrna = self.eliminate_low_mean_genes(df_mrna=df_mrna)
        df_mrna = np.log2(df_mrna + 1)
        df_cga =  df_cga[["Patient ID", "label", "Overall Survival (Months)"]].join(df_mrna, how="inner")
        df_cga = df_cga.dropna()
        df_cga = df_cga[df_cga["Overall Survival (Months)"] > 0]
        return df_cga
    
    def load_mrna(self) -> pd.DataFrame:
        """
        RSEM matrix (genes x samples) -> samples x genes, indexed by Sample ID
        """
        df_mrna = pd.read_csv(self.mrna_dataset, sep="\t").drop(columns=["Entrez_Gene_Id"])
        df_mrna = df_mrna.drop_duplicates(subset="Hugo_Symbol")  # shortcut: keeps first of 9 repeated symbols, revisit if one of them matters
        df_mrna = df_mrna.set_index("Hugo_Symbol").T.rename_axis("Sample ID")
        print(f"Shape of the mRNA (samples, genes): {df_mrna.shape}")
        return df_mrna
    
    def eliminate_zero_genes(self, df_mrna : pd.DataFrame, threshold : float = 0.01) -> pd.DataFrame:
        """
        Drop genes with zero counts in more than `threshold` of the samples (paper: 1%)
        """
        keep = (df_mrna == 0).mean(axis=0) <= threshold
        print(f"Zero filter (>{threshold:.0%} zeros): {df_mrna.shape[1]} -> {keep.sum()} genes")
        return df_mrna.loc[:, keep]
    
    def eliminate_low_mean_genes(self, df_mrna : pd.DataFrame, min_mean : float = 50) -> pd.DataFrame:
        """
        Drop genes whose mean RSEM count across samples is below `min_mean` (before log2)
        """
        keep = df_mrna.mean(axis=0) >= min_mean
        print(f"Mean filter (mean < {min_mean}): {df_mrna.shape[1]} -> {keep.sum()} genes")
        return df_mrna.loc[:, keep]
    
    def drop_columns(self, df_cga : pd.DataFrame) -> pd.DataFrame:
        df_cga = df_cga.drop(columns=self.COLUMNS_TO_DROP)
        df_cga = df_cga[df_cga["Sample Type"] == "Primary"]  # tumor sites only, as in the paper
        df_cga["label"] = df_cga["Overall Survival Status"].str.startswith("1").astype(int)  # 1 = deceased
        return df_cga
    
    
    def TextToCsv(self, file_path : str) -> pd.DataFrame:
        """
        Function for making text file to CSV format
        """
        df = pd.read_csv(file_path, sep="\\t", engine="python")
        print(f"Shape of the CSV: {df.shape}")
        return df
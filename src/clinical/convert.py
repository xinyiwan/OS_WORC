import pyreadstat

# Read without date parsing using the correct parameter name
df, meta = pyreadstat.read_sav(
    "/exports/lkeb-hpc/xwan/osteosarcoma/clinical_features/Osteosarcoma_data_langevelde_June24.sav",
    disable_datetime_conversion=True  # Correct parameter name
)

df.to_csv("/exports/lkeb-hpc/xwan/osteosarcoma/clinical_features/Osteosarcoma_data_langevelde_June24.csv", index=False)

key_columns = ['pat_nr', 'Age_Start', 'geslacht', 'pres_sympt', 'path_fract', 'Soft_Tissue_Exp', 'Size_primary_tumor',
               'Huvosnew', 'Huvos', 'Huvos_grading', 'Preop_protocl']
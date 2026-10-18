"""Human-readable schema, preserving original CDC codes at the raw-data boundary."""
import re
import unicodedata
import pandas as pd
from project import features, feature_names

NAMES = {
 'MAGER':'maternal_age_years','FAGECOMB':'paternal_age_years',
 'PRIORLIVE':'previous_births_now_living','PRIORDEAD':'previous_births_now_deceased',
 'PRIORTERM':'previous_other_pregnancy_terminations','CIG_0':'daily_cigarettes_before_pregnancy',
 'M_Ht_In':'maternal_height_inches','BMI':'prepregnancy_body_mass_index',
 'PWgt_R':'prepregnancy_weight_pounds','RF_CESARN':'previous_cesarean_count',
 'MRACEHISP':'maternal_race_ethnicity','DMAR':'maternal_marital_status',
 'MEDUC':'maternal_education','LBO_REC':'live_birth_order_category',
 'TBO_REC':'total_birth_order_category','RF_PDIAB':'prepregnancy_diabetes',
 'RF_PHYPE':'prepregnancy_hypertension','RF_PPTERM':'previous_preterm_birth',
 'RF_INFTR':'infertility_treatment','RF_FEDRG':'fertility_enhancing_drugs',
 'RF_ARTEC':'assisted_reproductive_technology','RF_CESAR':'previous_cesarean',
 'DPLURAL':'pregnancy_plurality','advanced_age':'maternal_age_at_least_35',
 'teen':'maternal_age_under_20','prior_preterm':'previous_preterm_birth_indicator',
 'prior_pregnancy_none':'no_previous_pregnancy_indicator','risk_count':'recorded_history_risk_count',
 'age_x_chronic_hypertension':'advanced_age_with_chronic_hypertension',
 'plurality_x_prior_preterm':'plurality_times_previous_preterm_indicator',
 'age_band':'maternal_age_group','bmi_category':'prepregnancy_bmi_category',
 'PREVIS':'total_prenatal_visits','PREVIS_REC':'prenatal_visit_count_category',
 'CIG_1':'daily_cigarettes_first_trimester','CIG_2':'daily_cigarettes_second_trimester',
 'CIG_3':'daily_cigarettes_third_trimester','WTGAIN':'pregnancy_weight_gain_pounds',
 'WTGAIN_REC':'pregnancy_weight_gain_category','RF_GDIAB':'gestational_diabetes',
 'RF_GHYPE':'gestational_hypertension','RF_EHYPE':'pregnancy_eclampsia',
 'WIC':'wic_receipt_recorded','IP_GON':'gonorrhea_recorded','IP_SYPH':'syphilis_recorded',
 'IP_CHLAM':'chlamydia_recorded','IP_HEPB':'hepatitis_b_recorded','IP_HEPC':'hepatitis_c_recorded',
 'OEGest_R3':'obstetric_gestation_three_category','GESTREC3':'alternative_gestation_three_category',
 'OEGest_Comb':'obstetric_gestation_weeks','COMBGEST':'alternative_gestation_weeks',
 'ILP_R':'recorded_interval_since_previous_pregnancy','PAY_REC':'delivery_payment_category',
 'SEX':'recorded_infant_sex','target':'preterm_birth','row_id':'source_record_id',
}

CODE_LABELS = {
 'MRACEHISP':{1:'Non-Hispanic White',2:'Non-Hispanic Black',3:'Non-Hispanic American Indian/Alaska Native',4:'Non-Hispanic Asian',5:'Non-Hispanic Native Hawaiian/Other Pacific Islander',6:'Non-Hispanic more than one race',7:'Hispanic',8:'Unknown/not stated'},
 'DMAR':{1:'Married',2:'Unmarried'},
 'MEDUC':{1:'8th grade or less',2:'9th-12th grade, no diploma',3:'High school/GED',4:'Some college, no degree',5:'Associate degree',6:'Bachelor degree',7:'Master degree',8:'Doctorate/professional degree',9:'Unknown'},
 'PAY_REC':{1:'Medicaid',2:'Private insurance',3:'Self pay',4:'Other',9:'Unknown'},
 'DPLURAL':{1:'Singleton',2:'Twins',3:'Triplets',4:'Quadruplet or higher'},
}

def code_legend():
    """Selected code meanings verified against supplied guide pp.16-17,32-33."""
    rows=[dict(source_code=field,readable_name=NAMES[field],code=str(value),meaning=label)
          for field,labels in CODE_LABELS.items() for value,label in labels.items()]
    for field in ['RF_PDIAB','RF_PHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR']:
        for value,label in [('Y','Yes'),('N','No'),('U','Unknown: converted to missing')]:
            rows.append(dict(source_code=field,readable_name=NAMES[field],code=value,meaning=label))
    for field in ['RF_FEDRG','RF_ARTEC']:
        rows.append(dict(source_code=field,readable_name=NAMES[field],code='X',meaning='Not applicable: preserved category'))
    return pd.DataFrame(rows)

def subgroup_label(variable,value):
    field={'race_ethnicity':'MRACEHISP','maternal_race_ethnicity':'MRACEHISP','insurance':'PAY_REC','delivery_payment_category':'PAY_REC','plurality':'DPLURAL','pregnancy_plurality':'DPLURAL','education':'MEDUC'}.get(variable)
    if field:
        try:code=int(float(value))
        except (ValueError,TypeError):return str(value)
        return CODE_LABELS[field].get(code,'Unknown / not reported' if code==-1 else str(value))
    return str(value)

def readable_feature_label(value):
    value=str(value)
    for code in sorted(NAMES,key=len,reverse=True):
        if value==code:return NAMES[code]
        for prefix in ['numeric__','categorical__']:
            if value==prefix+code or value.startswith(prefix+code+'_'):
                return value.replace(prefix+code,NAMES[code],1)
    return value

def readable_features(frame, engineered=True, time_features=False):
    """Reduce to approved inputs first, then rename; never expose targets to estimators."""
    return features(frame, engineered, time_features).rename(columns=NAMES)

def readable_names(engineered=True, time_features=False):
    numeric,categorical=feature_names(engineered,time_features)
    return [NAMES[n] for n in numeric],[NAMES[n] for n in categorical]

def build_readable_dictionary(dictionary):
    """Map every audited source column, including reporting-only/excluded columns."""
    result=dictionary.copy()
    def slug(text):
        text=unicodedata.normalize('NFKD',str(text)).encode('ascii','ignore').decode()
        return re.sub('[^a-z0-9]+','_',text.lower()).strip('_')
    result['readable_name']=[NAMES.get(r.field,slug(r.description)+'__'+r.field.lower()) for r in result.itertuples()]
    assert result.readable_name.is_unique
    nums,cats=feature_names()
    result['final_core_input']=result.field.isin(nums+cats)
    result['note_on_names']='Source codes preserved for CDC position verification; display/model aliases do not alter values.'
    return result

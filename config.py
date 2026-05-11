# keep all constants in one place

OUTPUT_PREFIX = 'legal_unlearning'

# model and dataset names
MODEL_NAME = 'Equall/Saul-7B-Instruct-v1'
NER_MODEL = 'dslim/distilbert-NER'
DATASET = 'joelniklaus/Multi_Legal_Pile_Commercial'
SUBSET = 'en_contracts'
LEGALBENCH = 'nguha/legalbench'

# NER and forget set configs
SAMPLE_SIZE = 500 # might increase this later
MIN_NAME_FREQ = 5 # minimum times a name needs to appear to be used in forget set
NER_SCORE_THRESHOLD = 0.90
MIN_NAME_TOKENS = 2
MAX_FORGET_PAIRS = 800 # might decrease this later
EVAL_FORGET_PAIRS = 200 # pairs for forget metrics
MAX_RETAIN_CHUNKS = 600
COMPLETION_MAX_TOKENS = 12

# NPO and retain hyperparameters
NPO_BETA = 0.1
NPO_LR = 5e-6
NPO_STEPS = 60
NPO_BATCH_SIZE = 4
MAX_SEQ_LEN = 256
RETAIN_BATCH_SIZE = 4
RETAIN_LOSS_WEIGHT = 1.0
EARLY_STOP_MIN_STEPS = 35 # minimum steps before npo loop can stop
EARLY_STOP_ENABLED = False
RETAIN_KL_SHIFTED = True
EARLY_STOP_RETAIN_KL = 0.12 # stop if retain KL spikes above this after min steps
EARLY_STOP_LOG_RATIO_TARGET = -40.0

# legal bench evaluation configs
LEGALBENCH_TASKS = {
    'hearsay',
    'personal_jurisdiction',
    'corporate_lobbying',
    'learned_hands_crime',
    'learned_hands_divorce'
}
MAX_SAMPLES_PER_TASK = 60
LEGALBENCH_MAX_LEN = 1024
NUM_FEW_SHOT = 3
# LEGALBENCH_MAX_LEN = 1024

SEED = 42

LOCATIONS = [
    'United States of America', 'American Samoa', 'Antigua and Barbuda',
    'Bosnia and Herzegovina', 'Brunei Darussalam', 'Burkina Faso', 'Cape Verde',
    'Cayman Islands', 'Central African Republic', 'Christmas Island', 'Cook Islands',
    'Czech Republic', 'Dominican Republic', 'East Timor', 'El Salvador', 'Equatorial Guinea',
    'Falkland Islands', 'Faroe Islands', 'French Guiana', 'French Polynesia', 'French Southern Territories',
    'The Gambia', 'Guinea-Bissau', 'Holy See', 'Hong Kong', 'Ivory Coast',
    'Marshall Islands', 'Republic of Maldova', 'Netherlands Antilles', 'New Caledonia',
    'New Zealand', 'North Macedonia', 'Northern Mariana Islands', 'Palestinian Territories',
    'Papua New Guinea', 'Pitcairn Island', 'Puerto Rico', 'Reunion Island', 'Russian Federation',
    'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'San Marino',
    'Sao Tome and Principe', 'Saudi Arabia', 'Sierra Leone', 'Solomon Islands',
    'South Africa', 'South Sudan', 'Sri Lanka', 'Syrian Arab Republic', 'Trinidad and Tobago',
    'Turks and Caicos Islands', 'United Arab Emirates', 'United Kingdom', 'United States',
    'Virgin Islands', 'Wallis and Futuna Islands', 'Western Sahara', 'North Korea', 'South Korea',
    'New York', 'New Jersey', 'North Dakota', 'South Dakota', 'North Carolina', 'South Carolina',
    'New Mexico', 'New Hampshire', 'Rhode Island', 'West Virginia', 'North America', 'South America',
    'Central America', 'Middle East', 'European Union'
]

LEGAL_TERMS = [
    'supreme court', 'district court', 'circuit court', 'federal union',
    'federal reserve', 'securities exchange', 'internal revenue'
]

ORG_SUFFIX = [
    'llc', 'inc', 'org', 'corp', 'corporation', 'company', 'co', 'plc', 'gmbh',
    'trust', 'fund', 'bank', 'partners', 'holdings', 's\.?a\.?', 'group', 'associates'
]

COMMON_WORDS = [
    'agreement', 'contract', 'section', 'article', 'general', 'council',
    'the', 'and', 'this', 'that', 'party', 'company'
]
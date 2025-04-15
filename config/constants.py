

major_img_modalities = ['CR', 'CT', 'MR', 'ES', 'US']

ALL_EVENT_TYPES = ('admission note', 'lab', 'imaging report', 'imaging exam', 'operative note', 'pathology report')

FEATURE_KEYS_CT = ('fmcib', 'SAMMed3D', 'universal-clipdriven')
FEATURE_KEYS_nonCT = ('SAMMed2D', )


FEATURE_DIMS = {
    'SAMMed2D': 256,
    'SAMMed3D': 384,
    'universal-clipdriven': 384,
    'fmcib': 4096,

    'ernie': 768,
    'pulse': 5120,

    'lab': 3 * 55,
}



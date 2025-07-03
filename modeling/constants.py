from enum import Enum


class EventTypes(Enum):
    BASIC_INFO = 'basic info'
    HOMEPAGE = '住院首页'
    ADMISSION = '出入院记录'
    CHEMO = '化疗'
    RADIATION = '放疗'
    LAB = '化验'
    IMG_REPORT = '影像报告'
    IMG_EXAM = '影像检查'
    SURGERY = '手术记录'
    PATHOLOGY = '病理报告'
    PATHODX = '病理诊断'
    WSI = 'WSI'


ALL_EVENT_TYPES = tuple(k.value for k in EventTypes if k != EventTypes.PATHODX)  # do not include PATHODX

COMPACT_EVENT_TYPES = ('出入院记录', '化验', '影像报告', '影像检查', '手术记录', '病理报告')

FEATURE_KEYS_CT = ('fmcib', 'SAMMed3D', 'universal-clipdriven')
FEATURE_KEYS_nonCT = ('SAMMed2D', )


FEATURE_DIMS_V3 = {
    'SAMMed2D': 256,
    'SAMMed3D': 384,
    'universal-clipdriven': 384,
    'fmcib': 4096,

    'ernie_content_only': 768,
    'pulse_content_only': 5120,

    'lab': 3 * 55,
}




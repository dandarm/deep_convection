"""Exploratory full-domain multispectral RDT-CW-inspired teacher."""
from .cloud_type_ir import IrCloudTypeConfig, SeviriIrCloudTypeTracker
from .config import RDTTeacherConfig
from .pipeline import RDTDomainTeacher, build_rdt_teacher_labels

__all__ = [
    "IrCloudTypeConfig",
    "SeviriIrCloudTypeTracker",
    "RDTTeacherConfig",
    "RDTDomainTeacher",
    "build_rdt_teacher_labels",
]

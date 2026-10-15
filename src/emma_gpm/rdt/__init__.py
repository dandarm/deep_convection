"""Exploratory full-domain multispectral RDT-CW-inspired teacher."""
from .config import RDTTeacherConfig
from .pipeline import RDTDomainTeacher, build_rdt_teacher_labels

__all__ = ["RDTTeacherConfig", "RDTDomainTeacher", "build_rdt_teacher_labels"]

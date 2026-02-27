"""
Simple typed payload models used by the dashboard layer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class School:
    id: int
    name: str


@dataclass
class Department:
    id: int
    school_id: int
    name: str
    vocational_field: str


@dataclass
class User:
    id: int
    email: str
    role: str
    school_id: Optional[int] = None


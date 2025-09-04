from typing import Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID

from .db import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = "users"
    full_name: Mapped[Optional[str]] = mapped_column(String(length=320), nullable=True)


